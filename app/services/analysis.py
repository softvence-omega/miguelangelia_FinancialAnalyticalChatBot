



# ============================================
# COMPLETE AUTOMATED SOLUTION
# ============================================
from ydata_profiling import ProfileReport
from autoviz.AutoViz_Class import AutoViz_Class
import sweetviz as sv
from app.services.llm_call import call_openai

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
                insights = call_openai(data_summary=)
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


