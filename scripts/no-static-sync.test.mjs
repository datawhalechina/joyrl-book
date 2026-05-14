import assert from 'node:assert/strict';
import {access, readFile} from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import {fileURLToPath} from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

async function pathMissing(relativePath) {
  await assert.rejects(access(path.join(repoRoot, relativePath)));
}

test('build scripts do not sync docs or notebooks into static', async () => {
  const packageJson = JSON.parse(await readFile(path.join(repoRoot, 'package.json'), 'utf8'));

  assert.equal(packageJson.scripts['sync-assets'], undefined);
  assert.equal(packageJson.scripts.prestart, undefined);
  assert.equal(packageJson.scripts.prebuild, undefined);
  await pathMissing('scripts/sync-static-assets.mjs');
});

test('static keeps only source-owned assets', async () => {
  await pathMissing('static/joyrl_docs');
  await pathMissing('static/llm_rl');
  await pathMissing('static/notebooks');
  await pathMissing('static/offline_rl');
  await pathMissing('static/rl_basic');
});
