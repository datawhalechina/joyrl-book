import React from 'react';
import Layout from '@theme-original/Layout';
import VisitorTracker from '@site/src/components/VisitorTracker';

type LayoutProps = React.ComponentProps<typeof Layout>;

export default function LayoutWrapper(props: LayoutProps): React.ReactElement {
  return (
    <Layout {...props}>
      <VisitorTracker />
      {props.children}
    </Layout>
  );
}
