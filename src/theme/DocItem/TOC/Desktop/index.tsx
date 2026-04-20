import React, {type ReactNode} from 'react';
import Link from '@docusaurus/Link';
import {ThemeClassNames} from '@docusaurus/theme-common';
import {useDoc} from '@docusaurus/plugin-content-docs/client';
import TOC from '@theme/TOC';
import {getInteractiveDocEntry} from '@site/src/components/interactive/interactiveRoutes';

function InteractiveModeButton({
  href,
}: {
  href: string;
}) {
  return (
    <Link
      to={href}
      aria-label="打开交互模式"
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: '0.9rem',
        padding: '0.78rem 0.9rem',
        borderRadius: '14px',
        textDecoration: 'none',
        color: '#ecfeff',
        background:
          'linear-gradient(135deg, rgba(15,118,110,0.96), rgba(8,145,178,0.96))',
        boxShadow: '0 18px 36px -24px rgba(8, 47, 73, 0.9)',
        fontSize: '0.92rem',
        fontWeight: 700,
        letterSpacing: '0.02em',
      }}>
      交互模式
    </Link>
  );
}

export default function DocItemTOCDesktop(): ReactNode {
  const {toc, frontMatter, metadata} = useDoc();
  const interactiveEntry = getInteractiveDocEntry(metadata.id);

  return (
    <div
      style={{
        position: 'sticky',
        top: 'calc(var(--ifm-navbar-height) + 2rem)',
      }}>
      {interactiveEntry ? (
        <InteractiveModeButton href={interactiveEntry.interactiveHref} />
      ) : null}
      <TOC
        toc={toc}
        minHeadingLevel={frontMatter.toc_min_heading_level}
        maxHeadingLevel={frontMatter.toc_max_heading_level}
        className={ThemeClassNames.docs.docTocDesktop}
      />
    </div>
  );
}
