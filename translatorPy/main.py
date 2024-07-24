from fastapi import FastAPI, HTTPException  # type: ignore
from pydantic import BaseModel # type: ignore
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer  # type: ignore
import torch  # type: ignore

app = FastAPI()

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

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