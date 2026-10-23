from pathlib import Path
import re

import pandas as pd
from transformers import pipeline


ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "raw" / "reviews.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "reviews_with_sentiment.csv"
MODEL_ID = "nlptown/bert-base-multilingual-uncased-sentiment"
MODEL_REVISION = "8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d"


def sentiment_band(stars: int) -> str:
	if stars <= 2:
		return "Negativo"
	if stars == 3:
		return "Neutral"
	return "Positivo"


def predict_reviews(reviews: list[str]) -> tuple[list[int], list[str]]:
	classifier = pipeline(
		"sentiment-analysis",
		model=MODEL_ID,
		revision=MODEL_REVISION,
	)
	predictions = classifier(reviews, batch_size=16, truncation=True)
	stars = []
	for prediction in predictions:
		match = re.search(r"\b([1-5])\b", prediction["label"])
		if match is None:
			raise ValueError(f"Unexpected model label: {prediction['label']!r}")
		stars.append(int(match.group(1)))
	return stars, [sentiment_band(value) for value in stars]


def main() -> None:
	reviews = pd.read_csv(INPUT_PATH)
	required_columns = {"review_id", "rating", "review_text"}
	missing_columns = required_columns.difference(reviews.columns)
	if missing_columns:
		raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
	if reviews["review_text"].isna().any() or reviews["review_text"].str.strip().eq("").any():
		raise ValueError("Every review must contain non-empty review_text")
	reviews["rating"] = pd.to_numeric(reviews["rating"], errors="raise")
	if not reviews["rating"].between(1, 5).all():
		raise ValueError("Human ratings must be between 1 and 5")

	stars, bands = predict_reviews(reviews["review_text"].astype(str).tolist())
	reviews["predicted_stars"] = stars
	reviews["predicted_sentiment"] = bands
	OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	reviews.to_csv(OUTPUT_PATH, index=False)
	print(f"Processed {len(reviews)} reviews: {OUTPUT_PATH}")


if __name__ == "__main__":
	main()
