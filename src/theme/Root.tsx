import React from 'react';
import VisitorTracker from '@site/src/components/VisitorTracker';

type RootProps = {
  children: React.ReactNode;
};

export default function Root({children}: RootProps): React.ReactElement {
  return (
    <>
      <VisitorTracker />
      {children}
    </>
  );
}
