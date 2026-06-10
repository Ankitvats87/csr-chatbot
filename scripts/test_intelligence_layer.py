import sys
import io
import asyncio
from pathlib import Path

# Reconfigure stdout to support UTF-8 characters like the Rupee symbol in Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.env import get_settings
from app.db.sqlite_client import SQLiteClient
from app.db.pinecone_client import PineconeClient
from app.services.embedding_service import EmbeddingService
from app.services.vector_service_v2 import VectorServiceV2
from app.services.response_service import ResponseService
from app.services.document_directory_service import DocumentDirectoryService
from app.services.intelligence_layer import IntelligenceLayerService
from app.models.message_model import Turn

async def test_scenarios():
    settings = get_settings()
    
    # Initialize database and clients
    sqlite = SQLiteClient(settings.sqlite_path)
    sqlite.connect()
    
    pinecone = PineconeClient(settings)
    pinecone.connect()
    
    embedder = EmbeddingService(settings)
    vectors = VectorServiceV2(pinecone, settings)
    responder = ResponseService(settings)
    directory = DocumentDirectoryService(sqlite)
    
    # Initialize intelligence service
    intel_svc = IntelligenceLayerService(
        embedder=embedder,
        vectors=vectors,
        responder=responder,
        directory=directory,
        settings=settings
    )
    
    print("=" * 80)
    print("TESTING MIDDLEWARE INTELLIGENCE LAYER")
    print("=" * 80)
    
    # Scenario 1: Project status & templates
    print("\n[SCENARIO 1: Project Status Query]")
    q1 = "What is the status of the Nirogya Life Line Foundation project?"
    print(f"Query: '{q1}'")
    # Call internal steps to print debug info
    plan1 = intel_svc._parse_plan(q1, [])
    print(f"  Rewritten Query: {plan1.get('rewritten_query')}")
    print(f"  Intent: {plan1.get('intent')}")
    print(f"  Extracted Entities: {plan1.get('entities')}")
    raw_proj = plan1.get('entities', {}).get('project_name')
    resolved_proj = intel_svc.resolve_project(raw_proj) if raw_proj else None
    print(f"  Resolved Project: {resolved_proj}")
    ans1 = await intel_svc.answer(chat_id=9999, question=q1, history=[])
    print(f"Answer:\n{ans1.answer}")
    print("-" * 80)
    
    # Scenario 2: Vague query handling
    print("\n[SCENARIO 2: Vague Query]")
    q2 = "Status?"
    print(f"Query: '{q2}'")
    plan2 = intel_svc._parse_plan(q2, [])
    print(f"  Rewritten Query: {plan2.get('rewritten_query')}")
    print(f"  Intent: {plan2.get('intent')}")
    print(f"  Extracted Entities: {plan2.get('entities')}")
    ans2 = await intel_svc.answer(chat_id=9999, question=q2, history=[])
    print(f"Answer:\n{ans2.answer}")
    print("-" * 80)
    
    # Scenario 3: Meeting summaries & templates
    print("\n[SCENARIO 3: Meeting Summary]")
    q3 = "Summarize the discussions and decisions in the 26th CSR meeting."
    print(f"Query: '{q3}'")
    plan3 = intel_svc._parse_plan(q3, [])
    print(f"  Rewritten Query: {plan3.get('rewritten_query')}")
    print(f"  Intent: {plan3.get('intent')}")
    print(f"  Extracted Entities: {plan3.get('entities')}")
    ans3 = await intel_svc.answer(chat_id=9999, question=q3, history=[])
    print(f"Answer:\n{ans3.answer}")
    print("-" * 80)
    
    # Scenario 4: Pronoun / History resolution
    print("\n[SCENARIO 4: Conversational History Resolution]")
    history = [
        Turn(role="user", content="What is the status of Swaroop project?"),
        Turn(role="assistant", content="Swaroop project is at Proposal stage withSwroop Charitable Foundation.")
    ]
    q4 = "Who is implementing it?"
    print("History: User: 'What is the status of Swaroop project?'")
    print(f"Query: '{q4}'")
    plan4 = intel_svc._parse_plan(q4, history)
    print(f"  Rewritten Query: {plan4.get('rewritten_query')}")
    print(f"  Intent: {plan4.get('intent')}")
    print(f"  Extracted Entities: {plan4.get('entities')}")
    raw_proj4 = plan4.get('entities', {}).get('project_name')
    resolved_proj4 = intel_svc.resolve_project(raw_proj4) if raw_proj4 else None
    print(f"  Resolved Project: {resolved_proj4}")
    ans4 = await intel_svc.answer(chat_id=9999, question=q4, history=history)
    print(f"Answer:\n{ans4.answer}")
    print("-" * 80)
    
    sqlite.close()

if __name__ == "__main__":
    asyncio.run(test_scenarios())
