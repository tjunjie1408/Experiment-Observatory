import { afterEach, describe, expect, it, vi } from "vitest";
import { BundleLoadError, loadRunBundle } from "../lib/loader";
import { makeBundle } from "./helpers/bundle";

function response(body: string, status = 200): Response {
  return new Response(body, { status });
}

function validResponses(): Response[] {
  const bundle = makeBundle();
  return [
    response(JSON.stringify(bundle.manifest)),
    response(bundle.events.map((event) => JSON.stringify(event)).join("\n")),
    response(JSON.stringify(bundle.snapshots)),
  ];
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("loadRunBundle", () => {
  it("distinguishes a missing artifact", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response("missing", 404)));

    await expect(loadRunBundle("/runs/missing")).rejects.toThrow(
      "manifest.json not found at /runs/missing/manifest.json (HTTP 404)",
    );
  });

  it("distinguishes malformed JSON and JSONL", async () => {
    const malformedJsonFetch = vi
      .fn()
      .mockResolvedValueOnce(response("{"))
      .mockResolvedValueOnce(response(""))
      .mockResolvedValueOnce(response("[]"));
    vi.stubGlobal("fetch", malformedJsonFetch);
    await expect(loadRunBundle("/runs/bad-json")).rejects.toThrow(
      "manifest.json is not valid JSON",
    );

    const bundle = makeBundle();
    const malformedJsonlFetch = vi
      .fn()
      .mockResolvedValueOnce(response(JSON.stringify(bundle.manifest)))
      .mockResolvedValueOnce(response("{\n"))
      .mockResolvedValueOnce(response(JSON.stringify(bundle.snapshots)));
    vi.stubGlobal("fetch", malformedJsonlFetch);
    await expect(loadRunBundle("/runs/bad-jsonl")).rejects.toThrow(
      "events.jsonl line 1 is not valid JSON",
    );
  });

  it("reports bundle validation failures", async () => {
    const responses = validResponses();
    const manifest = JSON.parse(await responses[0]!.text()) as Record<
      string,
      unknown
    >;
    manifest.schemaVersion = 999;
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValueOnce(response(JSON.stringify(manifest)))
        .mockResolvedValueOnce(responses[1])
        .mockResolvedValueOnce(responses[2]),
    );

    await expect(loadRunBundle("/runs/invalid")).rejects.toThrow(
      BundleLoadError,
    );
  });

  it("passes the abort signal to every artifact request", async () => {
    const controller = new AbortController();
    const responses = validResponses();
    const fetchMock = vi
      .fn()
      .mockImplementation((_url: string, init?: RequestInit) => {
        expect(init?.signal).toBe(controller.signal);
        return Promise.resolve(responses.shift());
      });
    vi.stubGlobal("fetch", fetchMock);

    await loadRunBundle("/runs/valid", controller.signal);

    expect(fetchMock).toHaveBeenCalledTimes(3);
  });
});
