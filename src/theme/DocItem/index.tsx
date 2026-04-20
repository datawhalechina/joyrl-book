import React, {type ReactNode, useEffect} from 'react';
import {HtmlClassNameProvider} from '@docusaurus/theme-common';
import {DocProvider} from '@docusaurus/plugin-content-docs/client';
import {useHistory, useLocation} from '@docusaurus/router';
import useBaseUrl from '@docusaurus/useBaseUrl';
import DocItemMetadata from '@theme/DocItem/Metadata';
import DocItemLayout from '@theme/DocItem/Layout';
import type {Props} from '@theme/DocItem';
import {
  getInteractiveDocEntry,
  isReadingModeSearch,
} from '@site/src/components/interactive/interactiveRoutes';

export default function DocItem(props: Props): ReactNode {
  const history = useHistory();
  const location = useLocation();
  const docHtmlClassName = `docs-doc-id-${props.content.metadata.id}`;
  const MDXComponent = props.content;
  const interactiveEntry = getInteractiveDocEntry(props.content.metadata.id);
  const interactiveHref = useBaseUrl(
    interactiveEntry?.interactiveHref ?? '/',
  );
  const shouldRedirect =
    Boolean(interactiveEntry) &&
    !isReadingModeSearch(location.search) &&
    !location.hash;

  useEffect(() => {
    if (shouldRedirect) {
      history.replace(interactiveHref);
    }
  }, [history, interactiveHref, shouldRedirect]);

  return (
    <DocProvider content={props.content}>
      <HtmlClassNameProvider className={docHtmlClassName}>
        <DocItemMetadata />
        <DocItemLayout>
          <MDXComponent />
        </DocItemLayout>
      </HtmlClassNameProvider>
    </DocProvider>
  );
}
