const DEFAULT_COUNTER_SHARDS = 64;
const DAILY_COUNTER_TTL_SECONDS = 60 * 60 * 24 * 14;
const BOT_USER_AGENT_PATTERN =
  /bot|crawler|spider|slurp|preview|headless|pingdom|curl|wget|python-requests|go-http-client/i;

function getConfiguredOrigins(env) {
  const rawOrigins = env.ALLOWED_ORIGINS || env.ALLOWED_ORIGIN || '';

  return rawOrigins
    .split(',')
    .map((origin) => origin.trim())
    .filter(Boolean);
}

function getAllowedOrigin(request, env) {
  const configuredOrigins = getConfiguredOrigins(env);
  const requestOrigin = request.headers.get('Origin');

  if (configuredOrigins.length === 0) {
    return requestOrigin || '*';
  }

  if (!requestOrigin) {
    return configuredOrigins[0];
  }

  if (configuredOrigins.includes(requestOrigin)) {
    return requestOrigin;
  }

  return null;
}

function buildCorsHeaders(allowedOrigin) {
  return {
    'Access-Control-Allow-Origin': allowedOrigin || 'null',
    'Access-Control-Allow-Credentials': 'true',
    'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Accept',
    'Access-Control-Max-Age': '86400',
    Vary: 'Origin',
  };
}

function json(data, init = {}, corsHeaders = {}) {
  const headers = new Headers(init.headers);
  headers.set('content-type', 'application/json; charset=UTF-8');
  headers.set('cache-control', 'no-store');

  Object.entries(corsHeaders).forEach(([key, value]) => {
    headers.set(key, value);
  });

  return new Response(JSON.stringify(data), {
    ...init,
    headers,
  });
}

function getTimezone(env) {
  return (env.TIMEZONE || 'Asia/Shanghai').trim() || 'Asia/Shanghai';
}

function getSiteId(env) {
  return (env.SITE_ID || 'joyrl-book').trim() || 'joyrl-book';
}

function getCounterShards(env) {
  const parsed = Number.parseInt(env.COUNTER_SHARDS || '', 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return DEFAULT_COUNTER_SHARDS;
  }
  return Math.min(parsed, 128);
}

function getDayKey(env) {
  const formatter = new Intl.DateTimeFormat('en-CA', {
    timeZone: getTimezone(env),
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  });

  return formatter.format(new Date());
}

function buildTotalKey(env, shardIndex) {
  return `${getSiteId(env)}:visits:total:shard:${shardIndex}`;
}

function buildDailyKey(env, dayKey, shardIndex) {
  return `${getSiteId(env)}:visits:day:${dayKey}:shard:${shardIndex}`;
}

function parseCounter(value) {
  const parsed = Number.parseInt(value || '0', 10);
  return Number.isFinite(parsed) ? parsed : 0;
}

function isRateLimitError(error) {
  return /429|rate limit/i.test(String(error));
}

function shouldSkipTracking(request) {
  if (request.method === 'HEAD') {
    return true;
  }

  const purpose =
    request.headers.get('purpose') ||
    request.headers.get('sec-purpose') ||
    request.headers.get('x-purpose') ||
    '';
  if (/prefetch|preview/i.test(purpose)) {
    return true;
  }

  const userAgent = request.headers.get('user-agent') || '';
  return BOT_USER_AGENT_PATTERN.test(userAgent);
}

async function incrementCounter(kv, key, options = {}) {
  const currentValue = parseCounter(await kv.get(key));
  await kv.put(key, String(currentValue + 1), options);
}

async function incrementShardedCounter(kv, buildKey, shardCount, options) {
  const triedShards = new Set();
  const maxAttempts = Math.min(shardCount, 8);

  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    let shardIndex = Math.floor(Math.random() * shardCount);

    while (triedShards.has(shardIndex) && triedShards.size < shardCount) {
      shardIndex = Math.floor(Math.random() * shardCount);
    }

    triedShards.add(shardIndex);

    try {
      await incrementCounter(kv, buildKey(shardIndex), options);
      return;
    } catch (error) {
      if (!isRateLimitError(error) || attempt === maxAttempts - 1) {
        throw error;
      }
    }
  }
}

async function readShardedCounter(kv, keys) {
  let total = 0;

  for (let index = 0; index < keys.length; index += 100) {
    const chunk = keys.slice(index, index + 100);
    const values = await kv.get(chunk);

    chunk.forEach((key) => {
      total += parseCounter(values.get(key));
    });
  }

  return total;
}

async function handleTrack(request, env, corsHeaders) {
  if (shouldSkipTracking(request)) {
    return json({ok: true, tracked: false, reason: 'skipped'}, {status: 202}, corsHeaders);
  }

  try {
    await request.clone().json();
  } catch {
    // The payload is optional for counting. We only care that the request succeeded.
  }

  const dayKey = getDayKey(env);
  const shardCount = getCounterShards(env);

  try {
    await Promise.all([
      incrementShardedCounter(
        env.VISITOR_STATS,
        (shardIndex) => buildTotalKey(env, shardIndex),
        shardCount,
      ),
      incrementShardedCounter(
        env.VISITOR_STATS,
        (shardIndex) => buildDailyKey(env, dayKey, shardIndex),
        shardCount,
        {expirationTtl: DAILY_COUNTER_TTL_SECONDS},
      ),
    ]);

    return json(
      {
        ok: true,
        tracked: true,
        dayKey,
      },
      {status: 202},
      corsHeaders,
    );
  } catch (error) {
    return json(
      {
        ok: false,
        error: 'Failed to persist visit counter',
        details: String(error),
      },
      {status: 500},
      corsHeaders,
    );
  }
}

async function handleStats(env, corsHeaders) {
  const dayKey = getDayKey(env);
  const shardCount = getCounterShards(env);
  const totalKeys = [];
  const todayKeys = [];

  for (let shardIndex = 0; shardIndex < shardCount; shardIndex += 1) {
    totalKeys.push(buildTotalKey(env, shardIndex));
    todayKeys.push(buildDailyKey(env, dayKey, shardIndex));
  }

  const [totalVisits, todayVisits] = await Promise.all([
    readShardedCounter(env.VISITOR_STATS, totalKeys),
    readShardedCounter(env.VISITOR_STATS, todayKeys),
  ]);

  return json(
    {
      todayVisits,
      totalVisits,
      dayKey,
      timezone: getTimezone(env),
      updatedAt: new Date().toISOString(),
      counterMode: 'pageviews',
    },
    {},
    corsHeaders,
  );
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const allowedOrigin = getAllowedOrigin(request, env);
    const configuredOrigins = getConfiguredOrigins(env);

    if (request.method === 'OPTIONS') {
      return new Response(null, {
        status: 204,
        headers: buildCorsHeaders(allowedOrigin || configuredOrigins[0] || '*'),
      });
    }

    if (request.headers.get('Origin') && !allowedOrigin) {
      return json(
        {
          ok: false,
          error: 'Origin not allowed',
        },
        {status: 403},
        buildCorsHeaders('null'),
      );
    }

    const corsHeaders = buildCorsHeaders(allowedOrigin || configuredOrigins[0] || '*');

    if (url.pathname === '/track' && request.method === 'POST') {
      return handleTrack(request, env, corsHeaders);
    }

    if (url.pathname === '/stats' && request.method === 'GET') {
      return handleStats(env, corsHeaders);
    }

    return json(
      {
        ok: true,
        name: 'joyrl-visitor-stats',
        endpoints: {
          stats: 'GET /stats',
          track: 'POST /track',
        },
      },
      {},
      corsHeaders,
    );
  },
};
