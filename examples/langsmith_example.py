#!/usr/bin/env python3
"""
Example demonstrating LangSmith integration with CogentNano.

This example shows how to use the LLM client with LangSmith tracing enabled.
Before running, ensure you have:
1. Set LANGSMITH_API_KEY in your .env file
2. Set OPENROUTER_API_KEY in your .env file
3. Set LANGSMITH_PROJECT in your .env file (optional)
"""

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from cogents.common.langsmith import get_langsmith_project, is_langsmith_enabled
from cogents.common.llm.openrouter import get_llm_client, get_llm_client_instructor


def test_basic_chat():
    """Test basic chat completion with LangSmith tracing."""
    print("🧪 Testing basic chat completion...")

    # Check if LangSmith is enabled
    if is_langsmith_enabled():
        print(f"✅ LangSmith tracing is enabled for project: {get_langsmith_project()}")
    else:
        print("⚠️  LangSmith tracing is not enabled")

    client = get_llm_client()

    messages = [
        {"role": "system", "content": "You are a helpful travel assistant."},
        {"role": "user", "content": "What are the top 3 attractions in Paris?"},
    ]

    response = client.chat_completion(messages=messages, temperature=0.7)
    print(f"🤖 Response: {response[:200]}...")

    return response


def test_structured_completion():
    """Test structured completion with LangSmith tracing."""
    print("\n🧪 Testing structured completion...")

    try:
        from typing import List

        from pydantic import BaseModel

        class Attraction(BaseModel):
            name: str
            description: str
            category: str

        class TravelRecommendations(BaseModel):
            city: str
            attractions: List[Attraction]
            best_time_to_visit: str

        client = get_llm_client_instructor()

        messages = [
            {
                "role": "system",
                "content": "You are a travel expert providing structured recommendations.",
            },
            {
                "role": "user",
                "content": "Give me travel recommendations for Tokyo with 3 top attractions.",
            },
        ]

        response = client.structured_completion(
            messages=messages, response_model=TravelRecommendations, temperature=0.7
        )

        print(f"🏙️  City: {response.city}")
        print(f"⏰ Best time to visit: {response.best_time_to_visit}")
        print("🎯 Attractions:")
        for attraction in response.attractions:
            print(f"  - {attraction.name}: {attraction.description}")

        return response

    except ImportError:
        print("❌ Pydantic not available for structured completion test")
        return None


def test_vision_capability():
    """Test vision understanding with LangSmith tracing."""
    print("\n🧪 Testing vision capability...")

    client = get_llm_client()

    # Test with a publicly available image URL
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
    prompt = "Describe what you see in this image."

    try:
        response = client.understand_image_from_url(image_url=image_url, prompt=prompt, temperature=0.7)
        print(f"👁️  Vision response: {response[:200]}...")
        return response
    except Exception as e:
        print(f"⚠️  Vision test failed: {e}")
        return None


def main():
    """Run all LangSmith integration tests."""
    print("🚀 Starting LangSmith integration tests for CogentNano")
    print("=" * 60)

    # Test basic chat
    test_basic_chat()

    # Test structured completion
    test_structured_completion()

    # Test vision capability
    test_vision_capability()

    print("\n" + "=" * 60)
    print("✨ LangSmith integration tests completed!")

    if is_langsmith_enabled():
        project = get_langsmith_project()
        print(f"📊 Check your traces at: https://smith.langchain.com/projects/{project}")
    else:
        print("💡 To enable LangSmith tracing:")
        print("   1. Get an API key from https://smith.langchain.com/")
        print("   2. Add LANGSMITH_API_KEY to your .env file")
        print("   3. Optionally set LANGSMITH_PROJECT for your project name")


if __name__ == "__main__":
    main()
