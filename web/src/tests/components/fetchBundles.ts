import { vi } from "vitest";
import type { RunBundle } from "../../lib/schema";

export interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
}

export function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

export function responseForBundle(url: string, bundle: RunBundle): Response {
  if (url.endsWith("manifest.json")) {
    return new Response(JSON.stringify(bundle.manifest));
  }
  if (url.endsWith("events.jsonl")) {
    return new Response(bundle.events.map((event) => JSON.stringify(event)).join("\n"));
  }
  if (url.endsWith("snapshots.json")) {
    return new Response(JSON.stringify(bundle.snapshots));
  }
  return new Response("Not found", { status: 404 });
}

export function mockBundleFetch(
  bundleForUrl: (url: string) => RunBundle | null,
): ReturnType<typeof vi.fn<typeof fetch>> {
  const fetchMock = vi.fn<typeof fetch>(async (input) => {
    const url = String(input);
    const bundle = bundleForUrl(url);
    return bundle === null
      ? new Response("Not found", { status: 404 })
      : responseForBundle(url, bundle);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}
