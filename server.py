from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import json
import asyncio
import base64
from io import BytesIO
from PIL import Image
import random

# Import Langchain integrations
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configuration
GEMINI_API_KEY = "AIzaSyC7VfNxfyqtESUEe92pf522gDomtjWsLIA" # Directly using the provided key

# Models
class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    message: str # Storing the user's message
    response: str # Storing the agent's response
    agent_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    message: str
    session_id: str
    language: Optional[str] = "english"

class PlantAnalysisRequest(BaseModel):
    session_id: str
    image_base64: str
    language: Optional[str] = "english"

class BudgetCalculationRequest(BaseModel):
    crop: str
    area_acres: float
    expected_yield_tons: float
    session_id: str

class LoanCalculationRequest(BaseModel):
    principal: float
    interest_rate: float
    tenure_years: int
    session_id: str

# Multi-Agent System
class AgriAssistAgents:
    def __init__(self):
        # Langchain models are initialized here, not in separate methods
        self.query_advisor_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
        self.plant_detection_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
        self.financial_advisor_model = ChatGoogleGenerativeAI( model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

    async def _get_chat_history_for_langchain(self, session_id: str, agent_type: str) -> List[Any]:
        """Fetches chat history for a specific session and agent type, formatted for Langchain."""
        history = []
        db_messages = await db.chat_messages.find(
            {"session_id": session_id, "agent_type": agent_type}
        ).sort("timestamp", 1).to_list(None) # Get all for the session

        for msg in db_messages:
            history.append(HumanMessage(content=msg['message']))
            history.append(AIMessage(content=msg['response']))
        return history

    async def get_query_advisor_response(self, session_id: str, message: str, language: str = "english") -> str:
        system_message_content = f"""You are an expert agricultural advisor specializing in Indian farming. 
        Respond in {language} language. 
        
        You provide:
        - Crop advice and schedules
        - Weather-related farming tips
        - General agricultural guidance
        - Pest and disease prevention
        - Soil management advice
        - Irrigation guidance
        - Seasonal farming recommendations
        
        Keep responses practical, farmer-friendly, and specific to Indian agricultural conditions.
        If asked about financial matters or plant diseases, suggest using the specialized tools available.
        
        IMPORTANT: Use simple formatting without asterisks (*) or complex markdown. Use numbered lists (1., 2., 3.) and simple bullet points (-) only. Avoid using ** for bold text.
        """
        
        # Fetch history for context
        history = await self._get_chat_history_for_langchain(session_id, "query_advisor")
        
        messages = [
            SystemMessage(content=system_message_content),
            *history,
            HumanMessage(content=message)
        ]
        
        response = await self.query_advisor_model.ainvoke(messages)
        return response.content
    
    async def get_plant_detection_response(self, session_id: str, image_base64: str, language: str = "english") -> str:
        system_message_content = f"""You are an expert plant pathologist and agricultural specialist.
        Respond in {language} language.
        
        Analyze plant images for:
        - Disease identification
        - Pest detection
        - Nutrient deficiencies
        - Plant health assessment
        
        For each analysis, provide:
        1. Detected issue with confidence percentage
        2. Organic treatment options
        3. Chemical treatment options
        4. Biological control methods
        5. Prevention strategies
        
        If confidence is below 70%, recommend consulting with agricultural extension officer.
        Keep recommendations practical and accessible to Indian farmers.
        
        IMPORTANT: Use simple formatting without asterisks (*) or complex markdown. Use numbered lists (1., 2., 3.) and simple bullet points (-) only. Avoid using ** for bold text.
        """
        
        # For gemini-pro-vision, content is a list of text and image parts
        # The image needs to be in a specific format for Langchain's HumanMessage
        image_data = base64.b64decode(image_base64)
        
        message_content = [
            {"type": "text", "text": "Analyze this plant image for diseases, pests, or health issues. Provide detailed diagnosis with confidence percentage and treatment recommendations."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ]

        # History is not typically used for single image analysis, but can be added if needed
        # For simplicity, not including chat history for vision model here
        
        messages = [
            SystemMessage(content=system_message_content),
            HumanMessage(content=message_content)
        ]
        
        response = await self.plant_detection_model.ainvoke(messages)
        return response.content
    
    async def get_financial_advisor_response(self, session_id: str, message: str, language: str = "english") -> str:
        system_message_content = f"""You are a financial advisor specializing in agricultural finance in India.
        Respond in {language} language.
        
        You provide:
        - Crop profitability analysis
        - Loan calculations and advice
        - Government scheme information
        - Market price insights
        - Insurance recommendations
        - Budget planning for farming
        
        Keep advice practical, include current government schemes, and focus on Indian agricultural markets.
        
        IMPORTANT: Use simple formatting without asterisks (*) or complex markdown. Use numbered lists (1., 2., 3.) and simple bullet points (-) only. Avoid using ** for bold text. Write clear, plain text responses.
        """
        
        # Fetch history for context
        history = await self._get_chat_history_for_langchain(session_id, "financial_advisor")
        
        messages = [
            SystemMessage(content=system_message_content),
            *history,
            HumanMessage(content=message)
        ]
        
        response = await self.financial_advisor_model.ainvoke(messages)
        return response.content

# Initialize agents
agents = AgriAssistAgents()

# Mock data for financial calculations
CROP_PRICES = {
    "tomato": {"cost_per_acre": 45000, "price_per_kg": 25},
    "onion": {"cost_per_acre": 35000, "price_per_kg": 18},
    "rice": {"cost_per_acre": 25000, "price_per_kg": 22},
    "wheat": {"cost_per_acre": 20000, "price_per_kg": 21},
    "sugarcane": {"cost_per_acre": 60000, "price_per_kg": 3},
    "cotton": {"cost_per_acre": 40000, "price_per_kg": 55},
}

GOVERNMENT_SCHEMES = [
    {
        "name": "PM-KISAN",
        "description": "₹6,000 per year direct benefit transfer",
        "eligibility": "All farmer families",
        "application": "Apply through PM-KISAN portal"
    },
    {
        "name": "Drip Irrigation Subsidy",
        "description": "50% subsidy on drip irrigation systems",
        "eligibility": "Small and marginal farmers",
        "application": "Apply through state agriculture department"
    },
    {
        "name": "Soil Health Card",
        "description": "Free soil testing and nutrient recommendations",
        "eligibility": "All farmers",
        "application": "Contact local agriculture extension office"
    },
    {
        "name": "KCC Loan",
        "description": "4% interest rate for crop loans",
        "eligibility": "Farmers with land documents",
        "application": "Apply through nearest bank"
    }
]

# API Endpoints
@api_router.post("/chat")
async def chat_with_advisor(request: ChatRequest):
    try:
        # Get response from query advisor
        response_content = await agents.get_query_advisor_response(request.session_id, request.message, request.language)
        
        # Save to database
        chat_message = ChatMessage(
            session_id=request.session_id,
            message=request.message,
            response=response_content,
            agent_type="query_advisor",
            metadata={"language": request.language}
        )
        
        await db.chat_messages.insert_one(chat_message.dict())
        
        return {"response": response_content, "agent_type": "query_advisor"}
        
    except Exception as e:
        logging.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/analyze-plant")
async def analyze_plant_image(request: PlantAnalysisRequest):
    try:
        # Get response from plant detection agent
        response_content = await agents.get_plant_detection_response(request.session_id, request.image_base64, request.language)
        
        # Extract confidence (mock for now, as direct confidence from LLM is not standard)
        confidence = random.randint(65, 95)
        
        # Save to database
        chat_message = ChatMessage(
            session_id=request.session_id,
            message="Plant image analysis", # User message for history
            response=response_content,
            agent_type="plant_detection",
            metadata={"language": request.language, "confidence": confidence}
        )
        
        await db.chat_messages.insert_one(chat_message.dict())
        
        return {
            "response": response_content,
            "confidence": confidence,
            "agent_type": "plant_detection",
            "escalate_to_officer": confidence < 70
        }
        
    except Exception as e:
        logging.error(f"Plant analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/calculate-budget")
async def calculate_crop_budget(request: BudgetCalculationRequest):
    try:
        # Get crop data
        crop_data = CROP_PRICES.get(request.crop.lower(), CROP_PRICES["tomato"])
        
        # Calculate costs and revenue
        total_cost = crop_data["cost_per_acre"] * request.area_acres
        expected_revenue = request.expected_yield_tons * 1000 * crop_data["price_per_kg"]  # Convert tons to kg
        profit = expected_revenue - total_cost
        
        analysis_prompt = f"""Analyze this crop budget:
        Crop: {request.crop}
        Area: {request.area_acres} acres
        Expected Yield: {request.expected_yield_tons} tons
        Total Cost: ₹{total_cost:,}
        Expected Revenue: ₹{expected_revenue:,}
        Profit: ₹{profit:,}
        
        Provide insights on profitability, risk factors, and recommendations.
        Use simple bullet points with - and numbered lists with 1., 2., 3. 
        Do not use any asterisks or markdown formatting in your response.
        """
        
        analysis_response = await agents.get_financial_advisor_response(request.session_id, analysis_prompt)
        
        # Save the interaction to the database
        chat_message = ChatMessage(
            session_id=request.session_id,
            message=analysis_prompt, # Store the prompt sent to the financial advisor
            response=analysis_response,
            agent_type="financial_advisor",
            metadata={"calculation_type": "budget", "crop": request.crop}
        )
        await db.chat_messages.insert_one(chat_message.dict())
        
        return {
            "crop": request.crop,
            "area_acres": request.area_acres,
            "expected_yield_tons": request.expected_yield_tons,
            "total_cost": total_cost,
            "expected_revenue": expected_revenue,
            "profit": profit,
            "analysis": analysis_response,
            "agent_type": "financial_advisor"
        }
        
    except Exception as e:
        logging.error(f"Budget calculation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/calculate-loan")
async def calculate_loan_emi(request: LoanCalculationRequest):
    try:
        # Calculate EMI
        monthly_rate = request.interest_rate / 12 / 100
        num_payments = request.tenure_years * 12
        
        # Handle zero interest rate case to avoid division by zero or incorrect formula
        if monthly_rate == 0:
            emi = request.principal / num_payments
        else:
            emi = (request.principal * monthly_rate * (1 + monthly_rate) ** num_payments) / ((1 + monthly_rate) ** num_payments - 1)
        
        # Optionally, get financial advisor's additional advice
        loan_advice_prompt = f"""A user has calculated a loan with the following details:
        Principal: ₹{request.principal:,}
        Interest Rate: {request.interest_rate}%
        Tenure: {request.tenure_years} years
        Monthly EMI: ₹{round(emi, 2):,}
        Total Amount Payable: ₹{round(emi * num_payments, 2):,}
        Total Interest: ₹{round((emi * num_payments) - request.principal, 2):,}

        Provide advice on managing this loan, suggest relevant government schemes, or financial planning tips for farmers in India regarding such a loan.
        Use simple bullet points with - and numbered lists with 1., 2., 3. 
        Do not use any asterisks or markdown formatting in your response.
        """
        
        loan_advice_response = await agents.get_financial_advisor_response(request.session_id, loan_advice_prompt)

        # Save the interaction to the database
        chat_message = ChatMessage(
            session_id=request.session_id,
            message=loan_advice_prompt, # Store the prompt sent to the financial advisor
            response=loan_advice_response,
            agent_type="financial_advisor",
            metadata={"calculation_type": "loan", "principal": request.principal, "interest_rate": request.interest_rate, "tenure_years": request.tenure_years}
        )
        await db.chat_messages.insert_one(chat_message.dict())

        return {
            "principal": request.principal,
            "interest_rate": request.interest_rate,
            "tenure_years": request.tenure_years,
            "monthly_emi": round(emi, 2),
            "total_amount": round(emi * num_payments, 2),
            "total_interest": round((emi * num_payments) - request.principal, 2),
            "financial_advice": loan_advice_response,
            "agent_type": "financial_advisor"
        }
        
    except Exception as e:
        logging.error(f"Loan calculation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/market-prices")
async def get_market_prices():
    try:
        # Mock market prices with random fluctuations
        prices = []
        for crop, data in CROP_PRICES.items():
            base_price = data["price_per_kg"]
            fluctuation = random.randint(-5, 5)  # ±5 rupees fluctuation
            current_price = base_price + fluctuation
            change = fluctuation
            
            prices.append({
                "crop": crop.title(),
                "price_per_kg": current_price,
                "change": change,
                "market": "Kochi Mandi"
            })
        
        return {"prices": prices}
        
    except Exception as e:
        logging.error(f"Market prices error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/government-schemes")
async def get_government_schemes():
    try:
        return {"schemes": GOVERNMENT_SCHEMES}
        
    except Exception as e:
        logging.error(f"Government schemes error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/weather-alerts")
async def get_weather_alerts():
    try:
        # Mock weather alerts
        alerts = [
            {
                "type": "warning",
                "message": "Heavy rain expected tomorrow",
                "action": "Protect crops and ensure proper drainage"
            },
            {
                "type": "info",
                "message": "Favorable weather for sowing",
                "action": "Good time for planting summer crops"
            }
        ]
        
        return {"alerts": alerts}
        
    except Exception as e:
        logging.error(f"Weather alerts error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/crop-advisory")
async def get_crop_advisory():
    try:
        # Mock seasonal advice
        advisory = [
            {
                "title": "Best for this season",
                "crops": ["Cucumber", "Beans", "Okra"],
                "reason": "Suitable for current weather conditions"
            },
            {
                "title": "High profit potential",
                "crops": ["Organic vegetables"],
                "reason": "30% premium in organic markets"
            },
            {
                "title": "Risk alert",
                "crops": ["Tomatoes"],
                "reason": "Late blight expected in humid conditions"
            }
        ]
        
        return {"advisory": advisory}
        
    except Exception as e:
        logging.error(f"Crop advisory error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/chat-history/{session_id}")
async def get_chat_history(session_id: str):
    try:
        messages = await db.chat_messages.find(
            {"session_id": session_id}
        ).sort("timestamp", -1).limit(50).to_list(50)
        
        return {"messages": [ChatMessage(**msg) for msg in messages]}
        
    except Exception as e:
        logging.error(f"Chat history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():

    client.close()



