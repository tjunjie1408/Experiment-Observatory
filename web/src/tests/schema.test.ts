import { describe, expect, it } from "vitest";
import { validateBundle } from "../lib/schema";
import { makeBundle, makeExternalBundle } from "./helpers/bundle";

function validateFixture(bundle: ReturnType<typeof makeBundle>) {
  return validateBundle(bundle.manifest, bundle.events, bundle.snapshots);
}

describe("validateBundle schema versions", () => {
  it("accepts a v1 manifest with the synthetic data configuration", () => {
    const bundle = makeBundle();

    expect(validateFixture(bundle).manifest.dataConfig).toEqual(
      bundle.manifest.dataConfig,
    );
  });

  it("accepts a v2 manifest with the fixed external-dataset configuration", () => {
    const bundle = makeExternalBundle();

    expect(
      validateBundle(bundle.manifest, bundle.events, bundle.snapshots).manifest
        .dataConfig,
    ).toEqual(bundle.manifest.dataConfig);
  });

  it("rejects data configuration shapes that do not match the manifest version", () => {
    const externalAsV1 = makeExternalBundle();
    externalAsV1.manifest.schemaVersion = 1;
    const syntheticAsV2 = makeBundle();
    syntheticAsV2.manifest.schemaVersion = 2;
    const hybridV2 = makeExternalBundle();
    Object.assign(
      hybridV2.manifest.dataConfig,
      makeBundle().manifest.dataConfig,
    );

    expect(() =>
      validateBundle(
        externalAsV1.manifest,
        externalAsV1.events,
        externalAsV1.snapshots,
      ),
    ).toThrow(/manifest\.dataConfig/);
    expect(() =>
      validateBundle(
        syntheticAsV2.manifest,
        syntheticAsV2.events,
        syntheticAsV2.snapshots,
      ),
    ).toThrow(/manifest\.dataConfig/);
    expect(() =>
      validateBundle(hybridV2.manifest, hybridV2.events, hybridV2.snapshots),
    ).toThrow(/manifest\.dataConfig/);
  });

  it.each([
    ["source", "synthetic"],
    ["sourceFeature", "horsepower"],
    ["feature", "weight"],
    ["featureUnit", "pounds"],
    ["target", "horsepower"],
    ["targetUnit", "kilometers per liter"],
    ["preprocessing", "none"],
    ["split", "train"],
  ])(
    "rejects a v2 external configuration with non-contract %s",
    (field, value) => {
      const bundle = makeExternalBundle();
      (bundle.manifest.dataConfig as unknown as Record<string, unknown>)[
        field
      ] = value;

      expect(() =>
        validateBundle(bundle.manifest, bundle.events, bundle.snapshots),
      ).toThrow(new RegExp(`manifest\\.dataConfig\\.${field}`));
    },
  );

  it("rejects unsupported manifest versions and event versions unequal to the manifest", () => {
    const unsupported = makeBundle();
    (unsupported.manifest as unknown as Record<string, unknown>).schemaVersion =
      3;
    const v2 = makeExternalBundle();
    v2.events.push({
      schemaVersion: 1,
      runId: v2.manifest.runId,
      seq: 0,
      kind: "run.created",
      step: null,
      message: null,
    });

    expect(() =>
      validateBundle(
        unsupported.manifest,
        unsupported.events,
        unsupported.snapshots,
      ),
    ).toThrow(/supports schema versions 1 and 2/);
    expect(() => validateBundle(v2.manifest, v2.events, v2.snapshots)).toThrow(
      /event schemaVersion 1 does not match manifest schemaVersion 2/,
    );
  });
});
