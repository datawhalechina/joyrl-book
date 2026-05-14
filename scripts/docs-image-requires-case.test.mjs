import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {readdir, readFile} from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import {fileURLToPath} from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const docsRoot = path.join(repoRoot, 'docs');

async function listMarkdownFiles(dir) {
  const entries = await readdir(dir, {withFileTypes: true});
  const files = [];

  for (const entry of entries) {
    const entryPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...await listMarkdownFiles(entryPath));
    } else if (entry.name.endsWith('.md') || entry.name.endsWith('.mdx')) {
      files.push(entryPath);
    }
  }

  return files;
}

function trackedFiles() {
  return new Set(
    execFileSync('git', ['ls-files', '-z'], {cwd: repoRoot})
      .toString('utf8')
      .split('\0')
      .filter(Boolean),
  );
}

function lineNumberForIndex(content, index) {
  return content.slice(0, index).split('\n').length;
}

test('Markdown require image paths match filesystem casing exactly', async () => {
  const failures = [];
  const files = await listMarkdownFiles(docsRoot);
  const tracked = trackedFiles();
  const requirePattern = /require\(["']([^"']+)["']\)/g;

  for (const file of files) {
    const content = await readFile(file, 'utf8');
    for (const match of content.matchAll(requirePattern)) {
      const target = path.resolve(path.dirname(file), match[1]);
      const relativeTarget = path.relative(repoRoot, target).split(path.sep).join('/');
      if (!tracked.has(relativeTarget)) {
        failures.push(`${path.relative(repoRoot, file)}:${lineNumberForIndex(content, match.index)} -> ${match[1]}`);
      }
    }
  }

  assert.deepEqual(failures, []);
});
