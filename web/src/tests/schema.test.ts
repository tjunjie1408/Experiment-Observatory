import { describe, expect, it } from "vitest";
import { validateBundle } from "../lib/schema";
import { makeBundle, makeExternalBundle } from "./helpers/bundle";

function validateFixture(bundle: ReturnType<typeof makeBundle>) {
  return validateBundle(bundle.manifest, bundle.events, bundle.snapshots);
}

describe("validateBundle schema versions", () => {
  it("accepts a v1 manifest with the synthetic data configuration", () => {
    const bundle = makeBundle();

    const validated = validateFixture(bundle);

    expect(validated.manifest.dataConfig).toEqual(bundle.manifest.dataConfig);
    expect(validated.manifest.dataset).toMatchObject({
      generatorId: "synthetic_linear",
    });
    expect(validated.manifest.dataset).not.toHaveProperty("sourceId");
  });

  it("accepts a v2 manifest with the fixed external-dataset configuration", () => {
    const bundle = makeExternalBundle();

    const validated = validateBundle(
      bundle.manifest,
      bundle.events,
      bundle.snapshots,
    );

    expect(validated.manifest.dataConfig).toEqual(bundle.manifest.dataConfig);
    expect(validated.manifest.dataset).toMatchObject({ sourceId: "auto-mpg" });
    expect(validated.manifest.dataset).not.toHaveProperty("generatorId");
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

  it("rejects dataset summary shapes that do not match the manifest version", () => {
    const external = makeExternalBundle();
    Object.assign(external.manifest.dataset, {
      generatorId: "synthetic_linear",
    });
    const synthetic = makeBundle();
    Object.assign(synthetic.manifest.dataset, { sourceId: "auto-mpg" });

    expect(() =>
      validateBundle(external.manifest, external.events, external.snapshots),
    ).toThrow(/manifest\.dataset.*generatorId/);
    expect(() =>
      validateBundle(synthetic.manifest, synthetic.events, synthetic.snapshots),
    ).toThrow(/manifest\.dataset.*sourceId/);
  });

  it.each([
    ["versionManifestSha256", "a".repeat(63)],
    ["versionManifestSha256", "A".repeat(64)],
    ["versionManifestSha256", `${"a".repeat(63)}g`],
    ["processedArtifactSha256", "b".repeat(65)],
    ["processedArtifactSha256", "B".repeat(64)],
    ["processedArtifactSha256", `${"b".repeat(63)}z`],
  ])(
    "rejects a v2 %s that is not lowercase 64-character hex",
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
