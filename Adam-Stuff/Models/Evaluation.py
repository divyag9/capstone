def usefulnessScore(extracted_statements,statements_list):
    """Use number of matches/ (total number of guesses + number of missed)
    as  a  criteria to see how our statement extractor is performing"""
    useful_statements = statements_list.copy()
    num_matches = 0
    total_useful = len(useful_statements)
    num_guesses = len(extracted_statements)
    for guess in extracted_statements:
        for actual in  useful_statements:
            A = set(guess.lower().split())
            B = set(actual.lower().split())
            if A.issubset(B) or B.issubset(A):
                num_matches += 1
                useful_statements.remove(actual)
                break
    num_missed = len(useful_statements)
    return num_matches/(num_guesses + num_missed), total_useful, num_matches, num_missed, num_guesses