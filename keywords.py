# Import Rake to get keywords
from rake_nltk import Rake


def get_keywords(message):
    # Use Rake class from rake-nltk
    r = Rake()
    r.extract_keywords_from_text(message)
    # Get keywords with their corresponding scores
    results = r.get_ranked_phrases_with_scores()

    # Return the keywords
    return results
