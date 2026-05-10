// Browser-side image downsizer. HF Spaces' proxy has aggressive body-size
// limits and a 4K phone capture (~10 MB JPEG) reliably triggers a "Failed
// to fetch" with no helpful error. Resizing before upload keeps every
// request under ~500 KB regardless of the source camera.
//
// Behaviour:
//   - Loads the input File via createImageBitmap (preserves orientation).
//   - If the longer side ≤ maxDim, returns the original unchanged.
//   - Otherwise, draws into a canvas with proportional scaling, encodes
//     to JPEG at the given quality, and returns a fresh File with a
//     descriptive name.
//
// The function is async + non-throwing on its happy path. If anything
// fails, it logs to console and returns the original File so the user
// still sees their picture and the request is just bigger than ideal.

export async function resizeImageForUpload(
  input: File,
  opts: { maxDim?: number; quality?: number } = {},
): Promise<File> {
  const maxDim = opts.maxDim ?? 1280;
  const quality = opts.quality ?? 0.85;

  // Skip non-image inputs.
  if (!input.type.startsWith("image/")) return input;

  try {
    const bitmap = await createImageBitmap(input);
    const longer = Math.max(bitmap.width, bitmap.height);
    if (longer <= maxDim) {
      bitmap.close();
      return input;
    }
    const scale = maxDim / longer;
    const w = Math.round(bitmap.width * scale);
    const h = Math.round(bitmap.height * scale);

    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      bitmap.close();
      return input;
    }
    ctx.drawImage(bitmap, 0, 0, w, h);
    bitmap.close();

    const blob: Blob | null = await new Promise((res) =>
      canvas.toBlob(res, "image/jpeg", quality),
    );
    if (!blob) return input;

    const baseName = input.name.replace(/\.[^.]+$/, "");
    return new File([blob], `${baseName}-resized.jpg`, { type: "image/jpeg" });
  } catch (e) {
    console.warn("resizeImageForUpload: falling back to original due to error", e);
    return input;
  }
}
