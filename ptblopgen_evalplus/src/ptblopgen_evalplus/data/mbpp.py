# import wget

# from evalplus.data.utils import (
#     CACHE_DIR,
#     completeness_check,
#     get_dataset_metadata,
#     make_cache,
#     stream_jsonl,
# )

# MBPP_PLUS_VERSION = "v0.2.0"
# MBPP_OVERRIDE_PATH = os.environ.get("MBPP_OVERRIDE_PATH", None)


# def _ready_mbpp_plus_path(mini=False, noextreme=False, version="default") -> str:
#     assert mini is False, "Mini version of MBPP+ is not available yet."

#     if MBPP_OVERRIDE_PATH:
#         return MBPP_OVERRIDE_PATH

#     version = MBPP_PLUS_VERSION if version == "default" else version

#     url, plus_path = get_dataset_metadata("MbppPlus", version, mini, noextreme)
#     make_cache(url, plus_path)

#     return plus_path


def mbpp_serialize_inputs(task_id: str, inputs: list) -> list:
    task_id = int(task_id.split("/")[-1])

    if task_id == 115:
        return [[[list(item) for item in inp[0]]] for inp in inputs]
    elif task_id == 124:
        return [(str(inp[0]), str(inp[1])) for inp in inputs]
    elif task_id == 252:
        return [[str(inp[0])] for inp in inputs]

    return inputs


def mbpp_deserialize_inputs(task_id: str, inputs: list) -> list:
    task_id = int(task_id.split("/")[-1])
    if task_id in [
        2,
        116,
        132,
        143,
        222,
        261,
        273,
        394,
        399,
        421,
        424,
        429,
        470,
        560,
        579,
        596,
        616,
        630,
        726,
        740,
        744,
        809,
    ]:
        modified_inputs = [[tuple(lst) for lst in inp] for inp in inputs]

    elif task_id in [
        63,
        64,
        70,
        94,
        120,
        237,
        272,
        299,
        400,
        409,
        417,
        438,
        473,
        614,
        780,
    ]:
        modified_inputs = [
            [[tuple(lst) for lst in lst_lst] for lst_lst in inp] for inp in inputs
        ]

    elif task_id in [75, 413, 444, 753]:
        modified_inputs = [
            [[tuple(lst) for lst in inp[0]]] + [inp[1]] for inp in inputs
        ]

    elif task_id == 106 or task_id == 750:
        modified_inputs = [[inp[0]] + [tuple(inp[1])] for inp in inputs]

    elif task_id == 115:
        modified_inputs = [
            [
                [
                    set(item) if isinstance(item, list) and len(item) else {}
                    for item in inp[0]
                ]
            ]
            for inp in inputs
        ]

    elif task_id == 124:
        modified_inputs = [(float(inp[0]), complex(inp[1])) for inp in inputs]

    elif task_id in [250, 405, 446, 617, 720, 763, 808]:
        modified_inputs = [[tuple(inp[0])] + [inp[1]] for inp in inputs]

    elif task_id in [259, 401, 445]:
        modified_inputs = [
            [[tuple(lst) for lst in lst_lst] for lst_lst in inp] for inp in inputs
        ]
        modified_inputs = [[tuple(lst) for lst in inp] for inp in modified_inputs]

    elif task_id == 278:
        modified_inputs = [
            [[tuple(item) if isinstance(item, list) else item for item in inp[0]]]
            for inp in inputs
        ]
        modified_inputs = [[tuple(lst) for lst in inp] for inp in modified_inputs]

    elif task_id == 307:
        modified_inputs = [[tuple(inp[0])] + [inp[1], inp[2]] for inp in inputs]

    elif task_id == 722:
        modified_inputs = [
            [{key: tuple(value) for key, value in inp[0].items()}] + inp[1:]
            for inp in inputs
        ]

    elif task_id == 252:
        modified_inputs = [[complex(inp[0])] for inp in inputs]

    elif task_id in [580, 615, 791]:

        def turn_all_list_into_tuple(inp):
            if isinstance(inp, list):
                return tuple([turn_all_list_into_tuple(item) for item in inp])
            return inp

        modified_inputs = [turn_all_list_into_tuple(inp) for inp in inputs]

    else:
        modified_inputs = inputs

    return modified_inputs
