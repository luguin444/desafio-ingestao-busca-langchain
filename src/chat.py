from search import search_prompt


def main():
    print('Chat RAG — Pergunte sobre o PDF ingerido. Digite "sair" para encerrar.\n')

    while True:
        try:
            question = input("PERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"sair", "exit", "quit"}:
            break

        try:
            answer = search_prompt(question)
        except Exception as e:
            print(f"[erro] {e}\n")
            continue

        print(f"RESPOSTA: {answer}\n")


if __name__ == "__main__":
    main()
