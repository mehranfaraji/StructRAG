import json
import time
import argparse
import random
from pathlib import Path

random.seed(1024)

from router import Router
from structurizer import Structurizer
from utilizer import Utilizer


def build_llm(args):
    if args.llm_name.lower() == "gemini":
        from utils.gemini_api import GeminiAPI

        return GeminiAPI(model=args.gemini_model, api_key=args.gemini_api_key)

    from utils.qwenapi import QwenAPI

    base_url = f"http://{args.url}/v1/chat/completions"
    return QwenAPI(url=base_url)


def build_router_llm(args, main_llm):
    if args.llm_name.lower() == "gemini" or args.router_url is None:
        return main_llm

    from utils.qwenapi import QwenAPI

    return QwenAPI(url=f"http://{args.router_url}/v1/chat/completions")


def load_loong_dataset(args):
    eval_data_path = "./Loong/data/loong_process.jsonl"
    eval_datas = [json.loads(l) for l in open(eval_data_path)]
    random.shuffle(eval_datas)
    eval_datas = eval_datas[200 * args.worker_id + args.start_bias: 200 * (args.worker_id + 1)]
    return eval_datas


def _extract_first_question(sample):
    questions = sample.get("questions") or []
    if isinstance(questions, dict):
        questions = [questions]
    for question in questions:
        if not isinstance(question, dict):
            continue
        text = question.get("question") or question.get("text") or question.get("query")
        if text:
            return text.strip()
    return None


def load_narrativeqa_dataset(args):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required to load the NarrativeQA dataset. "
            "Install it with `pip install datasets`."
        ) from exc

    split = args.narrativeqa_split
    limit = args.narrativeqa_limit
    dataset = load_dataset(
        "deepmind/narrativeqa",
        split=f"{split}[:{limit}]",
        token=args.hf_token,
    )

    eval_datas = []
    existing_ids = set()
    for idx, sample in enumerate(dataset):
        document = sample.get("document") or {}
        base_id = sample.get("id") or document.get("id") or f"narrativeqa-{split}-{idx}"
        doc_id = str(base_id)
        if doc_id in existing_ids:
            doc_id = f"{base_id}-{idx}"
        existing_ids.add(doc_id)

        title_candidates = [
            document.get("title"),
            document.get("url"),
            sample.get("story_title"),
            str(base_id),
        ]
        title = next((t for t in title_candidates if t), f"NarrativeQA Story {idx}")

        text_candidates = [
            document.get("text"),
            document.get("story"),
            document.get("summary"),
            sample.get("story"),
            sample.get("summary"),
        ]
        text = next((t for t in text_candidates if t), None)
        if text is None:
            raise ValueError(f"No narrative text found for NarrativeQA sample with id {base_id}.")
        text = text.strip()

        summary = document.get("summary") or sample.get("summary")

        docs_parts = [(title, text)]
        if summary and summary.strip() and summary.strip() != text:
            docs_parts.append((f"{title} Summary", summary.strip()))

        docs = "".join(
            f"<标题起始符>{part_title}<标题终止符>{part_text.strip()}" for part_title, part_text in docs_parts
        )

        question = _extract_first_question(sample)
        if question is None:
            question = f"Summarize the main events described in '{title}'."

        eval_datas.append({
            "id": str(doc_id),
            "instruction": "Use the provided narrative documents to answer the question in detail.",
            "question": question,
            "docs": docs,
            "prompt_template": "{instruction}\n\nQuestion:\n{question}\n\nDocuments:\n{docs}",
            "level": sample.get("set", "narrativeqa"),
            "set": split,
            "type": document.get("kind", "story"),
        })

    return eval_datas


def load_dataset_entries(args):
    if args.dataset_name.lower() == "narrativeqa":
        return load_narrativeqa_dataset(args)
    return load_loong_dataset(args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name", type=str, default="gemini")
    parser.add_argument("--dataset_name", type=str, default="narrativeqa")
    parser.add_argument("--url", type=str, default="10.32.15.63:1225")
    parser.add_argument("--router_url", type=str, default=None)
    parser.add_argument("--worker_id", type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7], default=0)
    parser.add_argument("--start_bias", type=int, default=0) # used to manually skip last time error data
    parser.add_argument("--output_path_suffix", type=str, default="")
    parser.add_argument("--gemini_model", type=str, default="gemini-1.5-flash-latest")
    parser.add_argument("--gemini_api_key", type=str, default=None)
    parser.add_argument("--narrativeqa_split", type=str, default="train")
    parser.add_argument("--narrativeqa_limit", type=int, default=3)
    parser.add_argument("--hf_token", type=str, default=None)
    args = parser.parse_args()

    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print('\nstart...')

    main_llm = build_llm(args)
    router_llm = build_router_llm(args, main_llm)

    eval_datas = load_dataset_entries(args)
    print(f"len eval_datas: {len(eval_datas)}")

    intermediate_results_dir = Path("./intermediate_results") / args.llm_name / f"{args.dataset_name}{args.output_path_suffix}"
    intermediate_results_dir.mkdir(parents=True, exist_ok=True)

    chunk_kb_path = intermediate_results_dir / "chunk_kb"
    graph_kb_path = intermediate_results_dir / "graph_kb"
    table_kb_path = intermediate_results_dir / "table_kb"
    algorithm_kb_path = intermediate_results_dir / "algorithm_kb"
    catalogue_kb_path = intermediate_results_dir / "catalogue_kb"
    for path in (chunk_kb_path, graph_kb_path, table_kb_path, algorithm_kb_path, catalogue_kb_path):
        path.mkdir(parents=True, exist_ok=True)

    output_dir = Path("./eval_results") / args.llm_name / f"{args.dataset_name}{args.output_path_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    success_path = output_dir / f"final_output_{args.worker_id}.jsonl"
    error_path = output_dir / f"final_output_error_{args.worker_id}.jsonl"
    if success_path.exists():
        with success_path.open("r", encoding="utf-8") as fr:
            existing_data = [json.loads(line) for line in fr if line.strip()]
    else:
        existing_data = []
    existing_data_ids = {d["id"] for d in existing_data}
    fw = success_path.open("a", encoding="utf-8")
    fw_error = error_path.open("a", encoding="utf-8")

    router = Router(router_llm)
    structurizer = Structurizer(
        main_llm,
        str(chunk_kb_path),
        str(graph_kb_path),
        str(table_kb_path),
        str(algorithm_kb_path),
        str(catalogue_kb_path),
    )
    utilizer = Utilizer(
        main_llm,
        str(chunk_kb_path),
        str(graph_kb_path),
        str(table_kb_path),
        str(algorithm_kb_path),
        str(catalogue_kb_path),
    )

    for i, data in enumerate(eval_datas): # data: {"instruction": "", "question": "", "docs": "", "prompt_template": "{},{},{}"}
        if data["id"] in existing_data_ids:
            print(f"################## Skipping {i}th data existing... ##################")
            continue
        print(f"################## Processing {i}th data... ##################")

        fw_intermediate = None
        try:
            current_time = time.time()
            fw_intermediate_path = intermediate_results_dir / f"{data['id']}.jsonl"
            fw_intermediate = fw_intermediate_path.open("w", encoding="utf-8")

            query = data['prompt_template'].format(instruction=data['instruction'], question=data['question'], docs="......")
            _, titles = structurizer.split_content_and_tile(data['docs'])
            core_content = "The titles of the docs are: " + "\n".join(list(set(titles)))

            # 1. router
            chosen = router.do_route(query, core_content, data['id'])  
            fw_intermediate.write(json.dumps({"query": query, "chosen": chosen}, ensure_ascii=False) + "\n")
            fw_intermediate.flush()

            # 2. structurizer
            instruction, kb_info = structurizer.construct(query, chosen, data['docs'], data['id'])
            fw_intermediate.write(json.dumps({"instruction": instruction, "kb_info": kb_info}, ensure_ascii=False) + "\n")
            fw_intermediate.flush()

            # 3. utilizer
            subqueries = utilizer.do_decompose(query, kb_info, data['id'])
            fw_intermediate.write(json.dumps({"subqueries": subqueries}, ensure_ascii=False) + "\n")
            fw_intermediate.flush()
            subknowledges = utilizer.do_extract(query, subqueries, chosen, data['id'])
            fw_intermediate.write(json.dumps({"subknowledges": subknowledges}, ensure_ascii=False) + "\n")
            fw_intermediate.flush()
            answer, _, _ = utilizer.do_merge(query, subqueries, subknowledges, chosen, data['id'])
            fw_intermediate.write(json.dumps({"answer": answer}, ensure_ascii=False) + "\n")
            fw_intermediate.flush()
            
            used_time = (time.time() - current_time) / 60
            print(f"level:{data['level']},set:{data['set']},type:{data['type']}")
            print(f"used time: {used_time:.2f} min")

            data['generate_response'] = answer
            data['used_time'] = used_time
            fw.write(json.dumps(data, ensure_ascii=False) + "\n")
            fw.flush()
            existing_data_ids.add(data["id"])
            fw_intermediate.close()

        except Exception as e:
            print(f"(print in main.py) Error: {e}")
            data['generate_response'] = "meet error"
            data['used_time'] = -100
            fw_error.write(json.dumps(data, ensure_ascii=False) + "\n")
            fw_error.flush()
            try:
                fw_intermediate.close()
            except Exception:
                pass

    fw.close()
    fw_error.close()
    print("all done")
