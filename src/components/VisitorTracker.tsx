import React, {useEffect, useRef} from 'react';
import {useLocation} from '@docusaurus/router';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

type CustomFields = {
  visitorStatsApiUrl?: string;
};

function normalizeApiBaseUrl(rawUrl: string): string {
  return rawUrl.endsWith('/') ? rawUrl.slice(0, -1) : rawUrl;
}

function normalizePagePath(pathname: string, baseUrl: string): string {
  if (!baseUrl || baseUrl === '/') {
    return pathname || '/';
  }

  const normalizedBaseUrl = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
  if (!normalizedBaseUrl || !pathname.startsWith(normalizedBaseUrl)) {
    return pathname || '/';
  }

  const strippedPath = pathname.slice(normalizedBaseUrl.length);
  return strippedPath || '/';
}

export default function VisitorTracker(): React.ReactElement | null {
  const {siteConfig} = useDocusaurusContext();
  const location = useLocation();
  const trackedPageRef = useRef('');
  const customFields = (siteConfig.customFields ?? {}) as CustomFields;
  const apiBaseUrl = normalizeApiBaseUrl((customFields.visitorStatsApiUrl ?? '').trim());
  const pagePath = normalizePagePath(location.pathname, siteConfig.baseUrl);

  useEffect(() => {
    if (!apiBaseUrl || typeof window === 'undefined') {
      return;
    }

    const pageKey = `${location.pathname}${location.search}`;
    if (trackedPageRef.current === pageKey) {
      return;
    }
    trackedPageRef.current = pageKey;

    const trackUrl = `${apiBaseUrl}/track`;
    const payload = JSON.stringify({
      path: pagePath,
      search: location.search,
      title: document.title,
      referrer: document.referrer || '',
    });

    const sendVisit = async (): Promise<void> => {
      if (navigator.sendBeacon) {
        try {
          const beaconBody = new Blob([payload], {type: 'application/json'});
          if (navigator.sendBeacon(trackUrl, beaconBody)) {
            return;
          }
        } catch {
          // Fall back to fetch when sendBeacon is unavailable or rejects the payload.
        }
      }

      try {
        await fetch(trackUrl, {
          method: 'POST',
          mode: 'cors',
          keepalive: true,
          headers: {
            'content-type': 'application/json',
          },
          body: payload,
        });
      } catch {
        // Ignore transient analytics failures so content rendering is never affected.
      }
    };

    void sendVisit();
  }, [apiBaseUrl, location.pathname, location.search, pagePath]);

  return null;
}
