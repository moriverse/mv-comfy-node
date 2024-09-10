import time

import execution
import server


class ExecutionTime:
    CATEGORY = "Moriverse/Debug"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "process"

    def process(self):
        return ()


CURRENT_START_EXECUTION_DATA = None


try:
    origin_execute = execution.execute

    def swizzle_execute(
        server,
        dynprompt,
        caches,
        current_item,
        extra_data,
        executed,
        prompt_id,
        execution_list,
        pending_subgraph_results,
    ):
        print("swizzle_execute..")
        unique_id = current_item
        class_type = dynprompt.get_node(unique_id)["class_type"]
        result = origin_execute(
            server,
            dynprompt,
            caches,
            current_item,
            extra_data,
            executed,
            prompt_id,
            execution_list,
            pending_subgraph_results,
        )

        print("swizzle_execute end..")

        global CURRENT_START_EXECUTION_DATA
        if not CURRENT_START_EXECUTION_DATA:
            return

        start_time = CURRENT_START_EXECUTION_DATA["nodes_start_perf_time"].get(
            unique_id
        )
        if start_time:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            print(
                f"[MVExecutionTime] NodeKey: {unique_id} - Node[{class_type}]: {execution_time:.2f}s"
            )

        return result

    execution.execute = swizzle_execute

except Exception as e:
    pass


try:
    origin_func = server.PromptServer.send_sync

    def swizzle_send_sync(self, event, data, sid=None):
        global CURRENT_START_EXECUTION_DATA
        if event == "execution_start":
            CURRENT_START_EXECUTION_DATA = dict(
                start_perf_time=time.perf_counter(), nodes_start_perf_time={}
            )

        print("prompt server send sync..")
        origin_func(self, event=event, data=data, sid=sid)
        print("prompt server send sync end..")

        if event == "executing" and data and CURRENT_START_EXECUTION_DATA:
            if data.get("node") is not None:
                CURRENT_START_EXECUTION_DATA["nodes_start_perf_time"][
                    data.get("node")
                ] = time.perf_counter()

    server.PromptServer.send_sync = swizzle_send_sync

except Exception as e:
    pass
