/**
 * Cloudflare Workers AI Embedding Provider
 *
 * 768 dimensions, using @cf/baai/bge-base-en-v1.5.
 * Adapted from Gemini provider to keep existing vector compatibility.
 */

const MODEL = "@cf/baai/bge-base-en-v1.5";

/**
 * Generate a text embedding via Cloudflare Workers AI.
 */
export async function getEmbedding(
  ai: Ai,
  text: string
): Promise<number[]> {
  const result = await ai.run(MODEL, { text: [text] });
  return (result as any).data[0];
}

/**
 * Generate an image embedding via Cloudflare Workers AI.
 * Workers AI doesn't support multimodal embeddings — falls back to text embedding of the description.
 */
export async function getImageEmbedding(
  ai: Ai,
  _imageData: ArrayBuffer,
  _mimeType: string,
  contextText?: string
): Promise<number[]> {
  return getEmbedding(ai, contextText || "image");
}
