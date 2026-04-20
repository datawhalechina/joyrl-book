import {cp, mkdir, readdir, rm} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const staticRoot = path.join(repoRoot, 'static');
const docsRoot = path.join(repoRoot, 'docs');
const notebooksRoot = path.join(repoRoot, 'notebooks');

const ignoredNames = new Set(['.DS_Store', '.nojekyll']);
const ignoredExtensions = new Set(['.md', '.mdx', '.html']);

function shouldSkip(name) {
  return name.startsWith('.') || ignoredNames.has(name);
}

function shouldCopyFile(name) {
  return !ignoredExtensions.has(path.extname(name).toLowerCase()) && !shouldSkip(name);
}

async function syncDocsAssets() {
  const entries = await readdir(docsRoot, {withFileTypes: true});

  for (const entry of entries) {
    if (shouldSkip(entry.name)) {
      continue;
    }

    const targetPath = path.join(staticRoot, entry.name);

    if (entry.isDirectory() || shouldCopyFile(entry.name)) {
      await rm(targetPath, {recursive: true, force: true});
    }
  }

  for (const entry of entries) {
    if (shouldSkip(entry.name)) {
      continue;
    }

    const sourcePath = path.join(docsRoot, entry.name);
    const targetPath = path.join(staticRoot, entry.name);

    if (entry.isDirectory()) {
      await cp(sourcePath, targetPath, {
        recursive: true,
        filter: (source) => shouldCopyFile(path.basename(source)) || !path.extname(source),
      });
      continue;
    }

    if (shouldCopyFile(entry.name)) {
      await cp(sourcePath, targetPath, {force: true});
    }
  }
}

async function syncNotebooks() {
  const targetRoot = path.join(staticRoot, 'notebooks');
  await rm(targetRoot, {recursive: true, force: true});
  await cp(notebooksRoot, targetRoot, {
    recursive: true,
    filter: (source) => {
      const name = path.basename(source);
      if (!path.extname(source)) {
        return !shouldSkip(name);
      }
      return shouldCopyFile(name);
    },
  });
}

await mkdir(staticRoot, {recursive: true});
await syncDocsAssets();
await syncNotebooks();

console.log('Synced docs assets into static/.');
