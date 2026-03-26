import base64
import io
from PIL import Image
from typing import Dict, List, Any
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2, ColQwen2Processor
import torch

class EndpointHandler():
    def __init__(self, path=""):
        self.model = ColQwen2.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",  # or "mps" if on Apple Silicon
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained(path) #, max_num_visual_tokens=8192) # temporary
        # self.model = torch.compile(self.model)
        print(f"Model and processor loaded {'with' if is_flash_attn_2_available() else 'without'} FA2")

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expects data in one of the following formats in the "inputs" key:
            {
                "images": [
                    "base64_encoded_image1",
                    "base64_encoded_image2",
                    ...
                ]
            }
            xor
            {
                "queries": [
                    "text1",
                    "text2",
                    ...
                ]
            }
        
        Returns embeddings for the provided input type.
        """
        # Input validation
        data = data.get("inputs", [])
        input_keys = [key for key in ["images", "queries"] if key in data]
        if len(input_keys) != 1:
            return {"error": "Exactly one of 'images', 'queries' must be provided"}
        
        input_type = input_keys[0]
        inputs = data[input_type]


        if input_type == "images":
            if not isinstance(inputs, list):
                inputs = [inputs]
        
            if len(inputs) > 8:
                return {"message": "Send a maximum of 8 images at once. We recommend sending one by one to improve load balancing."}

            # Decode each image from base64 and convert to a PIL Image
            decoded_images = []
            for img_str in inputs:
                try:
                    img_data = base64.b64decode(img_str)
                    image = Image.open(io.BytesIO(img_data)).convert("RGB")
                    decoded_images.append(image)
                except Exception as e:
                    return {"error": f"Error decoding image: {str(e)}"}

            # Process the images using the processor
            batch = self.processor.process_images(decoded_images).to(self.model.device)

        # elif input_type == "processed_images":
        #     try:
        #         buffer = io.BytesIO(base64.b64decode(inputs))
        #         batch = torch.load(buffer, map_location=self.model.device)
        #     except Exception as e:
        #         return {"error": f"Error processing preprocessed images: {str(e)}"}

        else:  # text
            if not isinstance(inputs, list):
                inputs = [inputs]
            try:
                batch = self.processor.process_queries(inputs).to(self.model.device)
            except Exception as e:
                return {"error": f"Error processing text: {str(e)}"}

        # Forward pass through the model
        with torch.inference_mode():
            embeddings = self.model(**batch).tolist()

        return {"embeddings": embeddings}
