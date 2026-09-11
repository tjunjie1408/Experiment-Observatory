/**
 * Loads a run bundle (manifest.json + events.jsonl + snapshots.json) from a
 * base URL and validates it. Distinguishes missing files, bad JSON, and
 * schema/consistency violations so the UI can show a specific reason
 * instead of a generic failure.
 */

import { BundleValidationError, type RunBundle, validateBundle } from "./schema";

export class BundleLoadError extends Error {
  constructor(reason: string) {
    super(reason);
    this.name = "BundleLoadError";
  }
}

async function fetchText(url: string, label: string): Promise<string> {
  let response: Response;
  try {
    response = await fetch(url);
  } catch (exc) {
    throw new BundleLoadError(`could not fetch ${label} (${url}): ${String(exc)}`);
  }
  if (!response.ok) {
    throw new BundleLoadError(`${label} not found at ${url} (HTTP ${response.status})`);
  }
  return response.text();
}

function parseJson(text: string, label: string): unknown {
  try {
    return JSON.parse(text);
  } catch (exc) {
    throw new BundleLoadError(`${label} is not valid JSON: ${String(exc)}`);
  }
}

function parseJsonl(text: string, label: string): unknown[] {
  const lines = text.split("\n").filter((line) => line.trim().length > 0);
  return lines.map((line, index) => {
    try {
      return JSON.parse(line);
    } catch (exc) {
      throw new BundleLoadError(`${label} line ${index + 1} is not valid JSON: ${String(exc)}`);
    }
  });
}

/**
 * Load and validate a run bundle from `baseUrl` (a directory containing
 * manifest.json, events.jsonl, snapshots.json). Supports an AbortSignal so
 * a caller switching to a different run can cancel an in-flight load: a
 * late-arriving response for an old selection must not overwrite a newer
 * selection.
 */
export async function loadRunBundle(baseUrl: string, signal?: AbortSignal): Promise<RunBundle> {
  const normalizedBase = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;

  const [manifestText, eventsText, snapshotsText] = await Promise.all([
    fetchText(`${normalizedBase}manifest.json`, "manifest.json"),
    fetchText(`${normalizedBase}events.jsonl`, "events.jsonl"),
    fetchText(`${normalizedBase}snapshots.json`, "snapshots.json"),
  ]);

  if (signal?.aborted) {
    throw new BundleLoadError("load cancelled");
  }

  const manifestRaw = parseJson(manifestText, "manifest.json");
  const eventsRaw = parseJsonl(eventsText, "events.jsonl");
  const snapshotsRawParsed = parseJson(snapshotsText, "snapshots.json");
  if (!Array.isArray(snapshotsRawParsed)) {
    throw new BundleLoadError("snapshots.json must contain a JSON array");
  }

  try {
    return validateBundle(manifestRaw, eventsRaw, snapshotsRawParsed);
  } catch (exc) {
    if (exc instanceof BundleValidationError) {
      throw new BundleLoadError(exc.message);
    }
    throw exc;
  }
}
