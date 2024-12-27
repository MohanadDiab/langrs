from langrs.core import LangRS

def main():
  text_input = "roof"

  image_input = "data/roi_kala.tif"

  langrs = LangRS(image_input, text_input, "output")

  langrs.predict_dino(window_size=600, overlap=300, box_threshold=0.25, text_threshold=0.25)

  langrs.outlier_rejection()

  langrs.predict_sam(rejection_method="zscore")

if __name__ == "__main__":
  main()