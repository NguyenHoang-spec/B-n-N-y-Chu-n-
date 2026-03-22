import { GoogleGenAI } from "@google/genai";

class GeminiEmbeddingService {
    private ai: GoogleGenAI | null = null;

    private getAI() {
        if (!this.ai) {
            // Initialize lazily to ensure API key is available
            this.ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
        }
        return this.ai;
    }

    /**
     * Generates a high-quality vector embedding using Gemini Cloud.
     * Model: gemini-embedding-2-preview (768 dimensions)
     */
    async embedText(text: string): Promise<number[]> {
        if (!text || text.trim().length === 0) return [];

        try {
            const ai = this.getAI();
            const result = await ai.models.embedContent({
                model: 'gemini-embedding-2-preview',
                contents: [text],
            });

            if (result.embeddings && result.embeddings.length > 0) {
                return result.embeddings[0].values;
            }
            return [];
        } catch (e: any) {
            console.error("Gemini Cloud Embedding failed:", e);
            
            // Graceful fallback for common errors
            if (e.message?.includes("quota") || e.message?.includes("429")) {
                console.warn("Quota exceeded for Gemini Embedding. Retrying might be needed.");
            }
            
            return [];
        }
    }

    // Alias for backward compatibility with existing code
    async embedTextLocal(text: string): Promise<number[]> {
        return this.embedText(text);
    }
}

export const localEmbeddingService = new GeminiEmbeddingService();
