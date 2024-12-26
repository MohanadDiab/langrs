from core.langrs import LangRS

def main():
  text_input = "roof"

  image_input = "data/roi_kala.tif"

  langrs = LangRS(image_input, text_input, "output")

  langrs.predict_dino(window_size=500, overlap=200)

  langrs.outlier_rejection()

  langrs.predict_sam()

if __name__ == "__main__":
  main()