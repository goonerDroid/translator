from fastapi import FastAPI, HTTPException  
from pydantic import BaseModel, field_validator 
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer  
import torch 
from typing import Any

app = FastAPI()

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

# List of supported languages by M2M100_418M
SUPPORTED_LANGUAGES = [
    "af", "am", "ar", "ast", "az", "ba", "be", "bg", "bn", "br", "bs", "ca", "ceb", "cs", "cy", "da", "de", "el", 
    "en", "es", "et", "fa", "ff", "fi", "fr", "fy", "ga", "gd", "gl", "gu", "ha", "he", "hi", "hr", "ht", "hu", 
    "hy", "id", "ig", "ilo", "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "lb", "lg", "ln", "lo", "lt", 
    "lv", "mg", "mk", "ml", "mn", "mr", "ms", "my", "ne", "nl", "no", "ns", "oc", "or", "pa", "pl", "ps", "pt", 
    "ro", "ru", "sd", "si", "sk", "sl", "so", "sq", "sr", "ss", "su", "sv", "sw", "ta", "th", "tl", "tn", "tr", 
    "uk", "ur", "uz", "vi", "wo", "xh", "yi", "yo", "zh", "zu"
]

class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

    @field_validator('source_lang', 'target_lang')
    @classmethod
    def validate_language(cls, lang: str, info: Any) -> str:
        if lang not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Language '{lang}' is not supported.",
                {
                    "field": info.field_name,
                    "supported_languages": SUPPORTED_LANGUAGES,
                    "total_supported": len(SUPPORTED_LANGUAGES)
                }
            )
        return lang

class TranslationResponse(BaseModel):
    text: str


@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    try:
        # Tokenize input text
        tokenizer.src_lang = request.source_lang
        encoded_text = tokenizer(request.text, return_tensors="pt")

        # Generate translation
        generated_tokens = model.generate(
            **encoded_text,
            forced_bos_token_id=tokenizer.get_lang_id(request.target_lang)
        )

        # Decode the generated tokens to text
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        return TranslationResponse(text=translation)
        
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


if __name__ == "__main__":
    import uvicorn  # type: ignore
    uvicorn.run(app, host="0.0.0.0", port=8000)