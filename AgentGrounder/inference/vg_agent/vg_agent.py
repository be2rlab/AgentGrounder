from typing import Any, Optional

from inference.embedding_database_class import create_embedding_database, EmbeddingDatabase

from langchain.agents import create_agent
from langchain_ollama.chat_models import ChatOllama
from inference.vg_agent.utils import format_messages, show_prompt
from inference.vg_agent.vg_agent_tools import  get_room_description, get_available_object_labels, get_ids_of_objects_with_label, calculate_distance_between_objects, find_farthest_object, list_top_k_nearest_objects, write_plan, get_object_info, get_object_info_by_labels, query_relevant_objects, list_top_k_nearest_objects_by_label, get_spatial_relationship, find_objects_within_radius, get_objects_by_vertical_position, find_nearest_object_to_group, find_nearest_pair_between_labels, see_image, see_image_with_object_ids
from inference.vg_agent.vg_agent_tools import FinalAnswer, RoomData
from langchain.agents.structured_output import ToolStrategy
from langchain.agents.middleware import TodoListMiddleware, LLMToolEmulator, SummarizationMiddleware, ModelCallLimitMiddleware
from langchain.agents.middleware import after_model, AgentState, hook_config, after_agent
from langchain.messages import HumanMessage, AIMessage
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from inference.vg_agent.vg_agent_tools import parse_final_answer, warn_call_limit

from inference.vg_agent.utils import load_bboxes, image_to_base64
from langchain_chroma import Chroma
from pathlib import Path
from langchain_ollama import OllamaEmbeddings

load_dotenv(dotenv_path='/home/docker_user/PCGrounder/configs/langsmith.env', override=True)

class VGAgent:
    def __init__(self, model, system_prompt: str, objects: dict, vectorstore: Optional[Chroma] = None):
        self.model = model
        self.room_data: RoomData = RoomData(objects=objects, vectorstore=vectorstore)
        
        self.agent = create_agent(
            model=self.model,
            system_prompt=system_prompt,
            response_format=ToolStrategy(schema=FinalAnswer),
            context_schema=RoomData,
            # tools=[get_ids_of_objects_with_label, list_top_k_nearest_objects, calculate_distance_between_objects, find_closest_object, find_farthest_object, write_plan, query_relevant_objects],
            # tools=[get_ids_of_objects_with_label, get_object_info, list_top_k_nearest_objects_by_label, get_spatial_relationship, find_objects_within_radius, get_objects_by_vertical_position, find_nearest_object_to_group, find_nearest_pair_between_labels, write_plan, query_relevant_objects, find_farthest_object, calculate_distance_between_objects, list_top_k_nearest_objects, see_image, see_image_with_object_ids],
            # tools=[write_plan, see_image],
            tools=[write_plan, get_object_info_by_labels, calculate_distance_between_objects, see_image_with_object_ids],
            # middleware=[TodoListMiddleware(), LLMToolEmulator(model=self.model)],
            middleware=[
                # check_for_final_answer,
                # SummarizationMiddleware(
                #     model=self.model,
                #     trigger=("fraction", 0.5),
                #     keep=("fraction", 0.3),
                # ),
                parse_final_answer,
                warn_call_limit,
                ModelCallLimitMiddleware(
                    thread_limit=None,
                    run_limit=10,
                    exit_behavior="end",
                ),
            ], # type: ignore
            # checkpointer=InMemorySaver()
        )


    def invoke(
        self,
        query,
        image_path: Optional[str],
        log_file: Optional[str] = None,
        scan_pc=None,
        center=None,
        render_save_dir: Optional[str] = None,
    ) -> FinalAnswer:
        # if image_path is not None:
        #     # Add image to HumanMessage
        #     image_base64 = image_to_base64(image_path)
        #     human_message = HumanMessage(content=[
        #         {"type": "text", "text": query},
        #         {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
        #     ])
        # else:
        self.room_data.image_path = image_path
        self.room_data.scan_pc = scan_pc
        self.room_data.center = center
        self.room_data.render_save_dir = render_save_dir

        human_message = HumanMessage(content=f"Please find the object in the room that best answers the following question: {query}")

        # image_base64 = image_to_base64(image_path)
        # image_message = HumanMessage(content=[
        #     {"type": "text", "text": "Please review the provided image and object 3D spatial descriptions, then select the object ID that best matches the given description."},
        #     {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
        # ])

        room_description_message = HumanMessage(content=get_room_description(objects=self.room_data.objects))
        available_object_labels_message = HumanMessage(content=get_available_object_labels(objects=self.room_data.objects))
        write_todo_message = HumanMessage(content="Now start by writing a detailed plan for how you will answer the user's question, including which tools you will use and in what order. You have to identify that if the query is view dependent, you should use the see_image tool. Then execute your plan, using the tools parallelly to gather information and make calculations as needed, until you arrive at a final answer to the user's query. Always provide a final answer in the specified structured format using the FinalAnswer schema, even if it is just a guess based on the information you have gathered.")

        response = self.agent.invoke(input={"messages": [human_message, room_description_message, available_object_labels_message, write_todo_message]}, context=self.room_data)

        format_messages(
            response["messages"],
            output_file=log_file,
            append=False,
            display=False,
        ) # save the conversation messages for debugging

        finalAnswer = response.get("structured_response", None)

        if finalAnswer is None:
            print("No structured response found in the agent's output. Returning a default FinalAnswer with object_id=-1.")
            return FinalAnswer(object_id=-1, explanation="No valid final answer found.")

        return finalAnswer

if __name__ == "__main__":
    model = ChatOllama(
        model="qwen3-vl:32b-instruct",
        # model="qwen3-vl:8b-instruct",
        # model="qwen3-vl:2b-instruct-q4_K_M",
        # model="qwen3-vl:8b-thinking",
        base_url="http://localhost:11435",
        num_ctx=8192,
        # profile={"max_input_tokens": 8192},
        temperature=0.0,
        # reasoning=True,
        # keep_alive=False
        )
    
    system_prompt = "You are a helpful and precise assistant for helping users find objects in a 3D scene. You are given a natural language query and a list of available tools that you can use to answer the query. You should use the tools to get information about the objects in the scene and their relationships, and then use that information to answer the user's question. Always think before acting. If you need to make calculations, write out the formula and calculate it step by step. If you need to compare distances, write out the distances and compare them explicitly. Be as detailed as possible in your reasoning steps, and make sure to use the tools effectively to gather all necessary information before answering the query. Then execute your plan step by step, using the tools to gather information and make calculations as needed, until you arrive at a final answer to the user's query. Always provide a final answer in the specified structured format using the FinalAnswer schema, even if it is just a guess based on the information you have gathered. If an image is provided, make sure to use it as part of your reasoning process and reference it in your answer if it helps you arrive at the correct answer. If there are objects with label 'object' in the room, make sure to check their metadata to see if they could be the object mentioned in the query, as sometimes objects may be mislabeled. Remember that the user is asking about objects in a 3D scene, so consider spatial relationships and distances between objects when reasoning about the answer. Please use parallell processing to call multiple tools at the same time when possible to gather information more efficiently. Always check if you have enough information to answer the question before making a guess, and if not, use the tools to gather more information until you are confident in your answer."


    room = "scene0011_00"
    olt_path = f"/home/huynh/repos/grounding/open_3dvg/SeeGround/data/nr3d/object_lookup_table/pred/{room}.json"

    objects = load_bboxes(olt_path)


    # embedder = OllamaEmbeddings(model="embeddinggemma:latest", base_url="http://localhost:11435")
    # db = EmbeddingDatabase(
    #     embedder=embedder,
    #     vectorstore_dir=Path("/home/huynh/repos/grounding/data/chromadb_data/v1"),
    #     collection_name=room,
    # )
    # results = db.vectorstore.similarity_search_with_relevance_scores("a door.", k=5, score_threshold=0.1)

    # print(results)

    image_path = "/home/huynh/repos/grounding/data/projection_img/scene0011_00/0/rendered.png"

    agent = VGAgent(model=model, system_prompt=system_prompt, objects=objects, vectorstore=None)

    query = "if you seeing the tv, which table is closest to the tv?"
    # image_path = "/home/huynh/repos/grounding/open_3dvg/rendered.png"


    finalAnswer: FinalAnswer = agent.invoke(query, image_path=image_path)
    print("Strucured final answer:", finalAnswer)