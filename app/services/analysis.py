"""
AI-Powered CSV to Dashboard Automation
Uses multiple AI libraries for automatic analysis and visualization
"""

# ============================================
# METHOD 1: Using PandasAI (AI-Powered Analysis)
# ============================================
"""
pip install pandasai
pip install matplotlib plotly
"""

# from pandasai import SmartDataframe
# from pandasai.llm import OpenAI
# import pandas as pd

# def method1_pandasai(csv_path, api_key):
#     """
#     PandasAI automatically analyzes CSV and generates insights
#     """
#     df = pd.read_csv(csv_path)
    
#     # Initialize AI model
#     llm = OpenAI(api_key=api_key)
#     smart_df = SmartDataframe(df, config={"llm": llm})
    
#     # AI automatically generates summary
#     summary = smart_df.chat("Give me a detailed summary of this data")
    
#     # AI creates visualizations
#     revenue_chart = smart_df.chat("Create a line chart showing revenue trends")
    
#     # AI finds insights
#     insights = smart_df.chat("What are the key insights from this data?")
    
#     return summary, insights


# ============================================
# METHOD 2: Using LlamaIndex (Document Analysis)
# ============================================
"""
pip install llama-index
pip install llama-index-readers-file
"""

# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
# from llama_index.readers.file import CSVReader

# def method2_llamaindex(csv_path):
#     """
#     LlamaIndex creates AI-powered data analysis
#     """
#     # Load CSV
#     parser = CSVReader()
#     documents = parser.load_data(file=csv_path)
    
#     # Create index for AI querying
#     index = VectorStoreIndex.from_documents(documents)
    
#     # Query engine
#     query_engine = index.as_query_engine()
    
#     # AI generates insights
#     summary = query_engine.query("Provide a comprehensive summary of this data")
#     insights = query_engine.query("What are the top 5 insights?")
#     trends = query_engine.query("What trends do you see?")
    
#     return summary, insights, trends


# # ============================================
# # METHOD 3: Using AutoViz (Automatic Visualization)
# ============================================
"""
pip install autoviz
"""

from autoviz.AutoViz_Class import AutoViz_Class

def method3_autoviz(csv_path):
    """
    AutoViz automatically creates all visualizations
    """
    AV = AutoViz_Class()
    
    # Automatically generate ALL charts
    dft = AV.AutoViz(
        filename=csv_path,
        sep=",",
        depVar="",  # target variable if any
        dfte=None,
        header=0,
        verbose=1,
        lowess=False,
        chart_format="html",  # Creates HTML output
        max_rows_analyzed=150000,
        max_cols_analyzed=30
    )
    
    return "autoviz_output.html"


# ============================================
# METHOD 4: Using Sweetviz (AI Report Generation)
# ============================================
"""
pip install sweetviz
"""

# import sweetviz as sv

# def method4_sweetviz(csv_path):
#     """
#     Sweetviz automatically generates beautiful analysis report
#     """
#     df = pd.read_csv(csv_path)
    
#     # Generate automatic report
#     report = sv.analyze(df)
#     report.show_html("sweetviz_report.html")
    
#     return "sweetviz_report.html"


# ============================================
# METHOD 5: Using D-Tale (Interactive Dashboard)
# ============================================
"""
pip install dtale
"""

# import dtale

# def method5_dtale(csv_path):
#     """
#     D-Tale creates automatic interactive dashboard
#     """
#     df = pd.read_csv(csv_path)
    
#     # Launch interactive dashboard
#     d = dtale.show(df)
    
#     # Export to HTML
#     d.to_html("dtale_dashboard.html")
    
#     return "dtale_dashboard.html"


# ============================================
# METHOD 6: Using Pandas Profiling (ydata-profiling)
# ============================================
"""
pip install ydata-profiling
"""

# from ydata_profiling import ProfileReport

# def method6_profiling(csv_path):
#     """
#     Pandas Profiling generates comprehensive analysis report
#     """
#     df = pd.read_csv(csv_path)
    
#     # Generate profile report
#     profile = ProfileReport(
#         df,
#         title="Automated Data Analysis Report",
#         explorative=True,
#         minimal=False
#     )
    
#     # Save to HTML
#     profile.to_file("profiling_report.html")
    
#     return "profiling_report.html"


# ============================================
# METHOD 7: Using Lux (Automatic Visualization)
# ============================================
"""
pip install lux-api
"""

# import lux

# def method7_lux(csv_path):
#     """
#     Lux automatically recommends and creates visualizations
#     """
#     df = pd.read_csv(csv_path)
    
#     # Lux automatically finds interesting visualizations
#     df  # Just display in Jupyter, Lux auto-generates charts
    
#     # Export visualizations
#     df.exported  # Get recommended visualizations
    
#     return df


# ============================================
# METHOD 8: Using OpenAI GPT for Analysis
# ============================================
"""
pip install openai
"""

import openai
import pandas as pd

def method8_openai(csv_path, api_key):
    """
    Use GPT-4 to analyze CSV and generate insights
    """
    openai.api_key = api_key
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Prepare data summary
    data_summary = f"""
    Columns: {df.columns.tolist()}
    Shape: {df.shape}
    Sample Data: {df.head(5).to_dict()}
    Statistics: {df.describe().to_dict()}
    """
    
    # Ask GPT to analyze
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a data analyst."},
            {"role": "user", "content": f"Analyze this data and provide insights:\n{data_summary}"}
        ]
    )
    
    insights = response.choices[0].message.content
    
    return insights


# ============================================
# COMPLETE AUTOMATED SOLUTION
# ============================================
from ydata_profiling import ProfileReport
from autoviz.AutoViz_Class import AutoViz_Class
import sweetviz as sv

class AIAutomatedDashboard:
    """
    Complete AI-powered automation using multiple tools
    """
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
    
    def generate_all(self, openai_api_key=None):
        """Generate everything automatically"""
        
        print("🤖 Starting AI-powered automation...")
        
        # 1. Automatic profiling report
        print("📊 Generating profiling report...")
        try:
            profile = ProfileReport(self.df, title="Auto Report", minimal=True)
            profile.to_file("auto_report.html")
            print("✅ Profiling report created: auto_report.html")
        except Exception as e:
            print(f"⚠️ Profiling failed: {e}")
        
        # 2. Automatic visualizations
        print("📈 Generating automatic visualizations...")
        try:
            AV = AutoViz_Class()
            AV.AutoViz(
                filename=self.csv_path,
                chart_format="html",
                verbose=0
            )
            print("✅ AutoViz charts created")
        except Exception as e:
            print(f"⚠️ AutoViz failed: {e}")
        
        # 3. Sweetviz report
        print("🍭 Generating Sweetviz report...")
        try:
            report = sv.analyze(self.df)
            report.show_html("sweetviz_dashboard.html", open_browser=False)
            print("✅ Sweetviz report created: sweetviz_dashboard.html")
        except Exception as e:
            print(f"⚠️ Sweetviz failed: {e}")
        
        # 4. AI-powered insights (if API key provided)
        if openai_api_key:
            print("🧠 Generating AI insights...")
            try:
                insights = method8_openai(self.csv_path, openai_api_key)
                with open("ai_insights.txt", "w") as f:
                    f.write(insights)
                print("✅ AI insights saved: ai_insights.txt")
            except Exception as e:
                print(f"⚠️ AI insights failed: {e}")
        
        print("\n🎉 Automation complete!")
        print("\nGenerated files:")
        print("  📄 auto_report.html - Comprehensive profiling")
        print("  📊 sweetviz_dashboard.html - Interactive dashboard")
        print("  📈 AutoViz charts - Multiple visualizations")
        if openai_api_key:
            print("  🧠 ai_insights.txt - AI-generated insights")


# ============================================
# USAGE EXAMPLE
# ============================================
from app.core.config import setting
if __name__ == "__main__":

    # Simple usage
    # dashboard = AIAutomatedDashboard("weatherHistory.csv")
    
    # # Generate everything automatically
    # dashboard.generate_all(openai_api_key=setting.openai_api_key)  # Optional
    
    # Or use individual methods:
    # method4_sweetviz("your_data.csv")  # Best for quick dashboards
    # method6_profiling("your_data.csv")  # Best for detailed analysis
    # method3_autoviz("your_data.csv")   # Best for visualizations

    import pandas as pd
    from ydata_profiling import ProfileReport

    df = pd.read_csv("weatherHistory.csv")
    profile = ProfileReport(df, title="Auto Dashboard")
    profile.to_file("dashboard.html")


"""
============================================
RECOMMENDATION:
============================================

🥇 BEST CHOICE: ydata-profiling (Pandas Profiling)
   - No API key needed
   - Automatic comprehensive report
   - Beautiful HTML dashboard
   - Installation: pip install ydata-profiling

🥈 SECOND BEST: Sweetviz
   - Very fast
   - Beautiful visualizations
   - Easy to use
   - Installation: pip install sweetviz

🥉 THIRD: AutoViz
   - Automatic chart generation
   - Multiple chart types
   - Installation: pip install autoviz

💡 FOR AI INSIGHTS: PandasAI or OpenAI
   - Requires API key
   - Natural language insights
   - Installation: pip install pandasai openai

============================================
QUICK START (No API Key Required):
============================================
"""


