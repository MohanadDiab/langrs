import rasterio
from samgeo import tms_to_geotiff, split_raster
from samgeo.text_sam import LangSAM
from samgeo import SamGeo
import rasterio
from rasterio.plot import show
import leafmap
from PIL import Image


class SamGeo():
    def __init__(self, image) -> None:
        self.sam = LangSAM()
        se
        pass

    def predict_dino(self,
                     image: Image = None,
                     prompt: str = None,
                     text_threshold: float = 0.1,
                     box_threshold: float = 0.1
                     ):
        
        if not image:
            raise KeyError("An image has to be provided")
        
        if not prompt:
            raise KeyError("A text prompt has to be provided")

        self.boxes, self.logits, self.phrases = self.sam.predict_dino(image = image,
                                        text_prompt = prompt,
                                        box_threshold = box_threshold,
                                        text_threshold = text_threshold
                                        )
        
        boxes = self.boxes
        
        

        pass

    def predict_sam(self):
        pass

    def remove_outliers(self):
        pass



        
        