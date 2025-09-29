import argparse
from VectorSpace import VectorSpace
from ParserZH import ParserZH
from Eval import load_smaller_dataset, evaluate_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Information Retrieval Assignment")
    parser.add_argument("--en_query", type=str, help="English query string")
    parser.add_argument("--feedback", action="store_true", help="Enable pseudo relevance feedback")
    parser.add_argument("--ch_query", type=str, help="Chinese query string")
    parser.add_argument("--eval", action="store_true", help="Run Task 4 evaluation")
    args = parser.parse_args()

    if not (args.en_query or args.ch_query or args.eval or args.feedback):
        parser.error("Please provide at least one of --en_query, --ch_query, or --eval.")

    if args.feedback and not args.en_query:
        parser.error("--feedback requires --en_query.")

    if args.en_query:
        vectorSpace = VectorSpace()
        vectorSpace.load_documents("EnglishNews")
        vectorSpace.build(vectorSpace.documents)

        resultswithcosineTF, resultswitheuclidean_s_TF, resultswitheuclidean_d_TF = vectorSpace.search(args.en_query.split(), use_tfidf=False)
        resultswithcosineTFIDF, resultswitheuclidean_s_TFIDF, resultswitheuclidean_d_TFIDF = vectorSpace.search(args.en_query.split(), use_tfidf=True)
        
        if not args.feedback:
            print("English TF Cosine")
            print(f"{'NewsID':<15}{'Score':>6}")
            for name, score in resultswithcosineTF:
                print(f"{name:<15}{score:>10.7f}")
            print("-"*40)

            print("English TF-IDF Cosine")
            print(f"{'NewsID':<15}{'Score':>6}")
            for name, score in resultswithcosineTFIDF:
                print(f"{name:<15}{score:>10.7f}")
            print("-"*40)

            print("English TF Euclidean")
            print(f"{'NewsID':<15}{'Score':>6}")
            for name, score in resultswitheuclidean_s_TF:
                print(f"{name:<15}{score:>10.7f}")
            print("-"*40)

            print("English TF-IDF Euclidean")
            print(f"{'NewsID':<15}{'Score':>6}")
            for name, score in resultswitheuclidean_d_TFIDF:
                print(f"{name:<15}{score:>10.7f}")
            print("-"*40)

        if args.feedback:
            print("English TF-IDF Cosine with Relevance Feedback")
            feedback_results = vectorSpace.feedback_research(args.en_query.split(), x=1.0, y=0.5)
            print(f"{'NewsID':<15}{'Score':>6}")
            for name, score in feedback_results:
                print(f"{name:<15}{score:>10.7f}")
            print("-"*40)

    if args.ch_query:
        vectorSpace = VectorSpace()
        vectorSpace.load_documents("ChineseNews")
        vectorSpace.parser = ParserZH(hmm=True)
        vectorSpace.build(vectorSpace.documents)

        results_TF, _, _ = vectorSpace.search(args.ch_query.split(), use_tfidf=False)
        results_TFIDF, _, _ = vectorSpace.search(args.ch_query.split(), use_tfidf=True)

        print("Chinese TF Cosine")
        print(f"{'NewsID':<15}{'Score':>6}")
        for name, score in results_TF[:10]:
            print(f"{name:<15}{score:>10.7f}")
        print("-"*40)

        print("Chinese TF-IDF Cosine")
        print(f"{'NewsID':<15}{'Score':>6}")
        for name, score in results_TFIDF[:10]:
            print(f"{name:<15}{score:>10.7f}")
        print("-"*40)
    
    if args.eval:
        doc_names, documents, quer_names, queries, relevant = load_smaller_dataset()
        mrr, map10, r10 = evaluate_all(doc_names, documents, quer_names, queries, relevant, 10, True)

        print("Evaluation IR system with TF-IDF Cosine")
        print(f"MRR@10         {mrr:.6f}")
        print(f"MAP@10         {map10:.6f}")
        print(f"RECALL@10      {r10:.6f}")
        print("-"*24)
