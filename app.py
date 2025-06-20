# app.py - Main Flask application for Render deployment
import os
import json
import tempfile
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging
from datetime import datetime
import zipfile
import io

# Import our resume analyzer classes
from resume_analyzer import ResumeJobMatcher, ResumeAnalysis, JobPlatformScraper

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = '/tmp'  # Use /tmp for temporary files on Render

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the matcher with environment variables
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
JOOBLE_API_KEY = os.environ.get('JOOBLE_API_KEY') # New: Get Jooble key

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY environment variable not set")
if not JOOBLE_API_KEY:
    logger.warning("JOOBLE_API_KEY environment variable not set. Job search will be disabled.")


matcher = None
job_scraper = None  # New: Standalone job scraper instance

if GROQ_API_KEY:
    try:
        # Pass both keys to the matcher
        matcher = ResumeJobMatcher(groq_api_key=GROQ_API_KEY, jooble_api_key=JOOBLE_API_KEY)
        logger.info("ResumeJobMatcher initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ResumeJobMatcher: {e}")

# Initialize standalone job scraper for manual search
if JOOBLE_API_KEY:
    try:
        job_scraper = JobPlatformScraper(jooble_api_key=JOOBLE_API_KEY)
        logger.info("JobPlatformScraper initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize JobPlatformScraper: {e}")

@app.route('/')
def home():
    """Home page with upload form"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "groq_api_configured": GROQ_API_KEY is not None,
        "jooble_api_configured": JOOBLE_API_KEY is not None # New
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_resume():
    """Main endpoint to analyze resume and find jobs"""
    if not matcher:
        return jsonify({"error": "Service not properly configured"}), 500
    
    try:
        # Check if file is present
        if 'resume' not in request.files:
            return jsonify({"error": "No resume file provided"}), 400
        
        file = request.files['resume']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Get optional parameters
        location = request.form.get('location', '')
        jobs_per_platform = int(request.form.get('jobs_per_platform', 10)) # Default to 10
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        try:
            # Analyze resume
            logger.info(f"Analyzing resume: {filename}")
            resume_analysis = matcher.analyze_resume_file(temp_path)
            
            # Find matching jobs
            logger.info("Finding matching jobs...")
            matched_jobs = matcher.find_relevant_jobs(
                resume_analysis, 
                location=location, 
                jobs_per_platform=jobs_per_platform
            )
            
            # Generate report
            report = matcher.generate_report(resume_analysis, matched_jobs)
            
            # Prepare response data
            response_data = {
                "analysis": {
                    "skills": resume_analysis.skills,
                    "experience_years": resume_analysis.experience_years,
                    "job_titles": resume_analysis.job_titles,
                    "education": resume_analysis.education,
                    "certifications": resume_analysis.certifications,
                    "keywords": resume_analysis.keywords,
                    "summary": resume_analysis.summary
                },
                "matched_jobs": [
                    {
                        "title": job_match["job"].title,
                        "company": job_match["job"].company,
                        "location": job_match["job"].location,
                        "platform": job_match["job"].platform,
                        "relevance_score": job_match["relevance_score"],
                        "salary": job_match["job"].salary,
                        "url": job_match["job"].url,
                        "description": job_match["job"].description[:200] + "..." if len(job_match["job"].description) > 200 else job_match["job"].description
                    }
                    for job_match in matched_jobs[:20]  # Limit to top 20 jobs
                ],
                "report": report,
                "total_jobs_found": len(matched_jobs),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Analysis completed successfully. Found {len(matched_jobs)} jobs.")
            return jsonify(response_data)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/api/search-jobs', methods=['POST'])
def manual_job_search():
    """New endpoint for manual job search without resume analysis"""
    if not job_scraper:
        return jsonify({"error": "Job search service not properly configured"}), 500
    
    try:
        data = request.get_json()
        
        # Extract search parameters
        keywords = data.get('keywords', '').strip()
        location = data.get('location', '').strip()
        job_limit = int(data.get('limit', 20))
        
        if not keywords:
            return jsonify({"error": "Keywords are required for job search"}), 400
        
        logger.info(f"Manual job search - Keywords: '{keywords}', Location: '{location}', Limit: {job_limit}")
        
        # Search for jobs
        jobs = job_scraper.search_jooble(keywords, location, job_limit)
        
        if not jobs:
            return jsonify({
                "jobs": [],
                "total_found": 0,
                "message": "No jobs found matching your criteria. Try different keywords or location.",
                "search_timestamp": datetime.now().isoformat()
            })
        
        # Format response
        job_results = [
            {
                "title": job.title,
                "company": job.company,
                "location": job.location,
                "platform": job.platform,
                "salary": job.salary,
                "url": job.url,
                "description": job.description[:300] + "..." if len(job.description) > 300 else job.description,
                "posted_date": job.posted_date
            }
            for job in jobs
        ]
        
        response_data = {
            "jobs": job_results,
            "total_found": len(jobs),
            "search_parameters": {
                "keywords": keywords,
                "location": location,
                "limit": job_limit
            },
            "search_timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Manual job search completed successfully. Found {len(jobs)} jobs.")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error during manual job search: {str(e)}")
        return jsonify({"error": f"Job search failed: {str(e)}"}), 500

@app.route('/api/download-report', methods=['POST'])
def download_report():
    """Download analysis report as text file"""
    try:
        data = request.get_json()
        report_content = data.get('report', '')
        
        if not report_content:
            return jsonify({"error": "No report content provided"}), 400
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        temp_file.write(report_content)
        temp_file.close()
        
        return send_file(
            temp_file.name,
            as_attachment=True,
            download_name=f'resume_analysis_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
            mimetype='text/plain'
        )
        
    except Exception as e:
        logger.error(f"Error generating report download: {str(e)}")
        return jsonify({"error": "Failed to generate report"}), 500

@app.route('/api/download-jobs', methods=['POST'])
def download_jobs():
    """New endpoint to download job search results as text file"""
    try:
        data = request.get_json()
        jobs_data = data.get('jobs', [])
        search_params = data.get('search_parameters', {})
        
        if not jobs_data:
            return jsonify({"error": "No jobs data provided"}), 400
        
        # Generate job search report
        report_content = f"""
JOB SEARCH RESULTS
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== SEARCH PARAMETERS ===
Keywords: {search_params.get('keywords', 'N/A')}
Location: {search_params.get('location', 'Any')}
Results Found: {len(jobs_data)}

=== JOB LISTINGS ===
"""
        
        for i, job in enumerate(jobs_data, 1):
            report_content += f"""
{i}. {job['title']} at {job['company']}
   Platform: {job['platform']}
   Location: {job['location']}
   Salary: {job['salary'] or 'Not specified'}
   Posted: {job['posted_date'] or 'Not specified'}
   URL: {job['url']}
   
   Description:
   {job['description']}
   
   {'='*50}
"""
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8')
        temp_file.write(report_content)
        temp_file.close()
        
        return send_file(
            temp_file.name,
            as_attachment=True,
            download_name=f'job_search_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
            mimetype='text/plain'
        )
        
    except Exception as e:
        logger.error(f"Error generating job search download: {str(e)}")
        return jsonify({"error": "Failed to generate job search report"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)