import os
import re
import json
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

# Set up logging
logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF for PDF processing
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    logger.warning("PyMuPDF not available - PDF processing disabled")

try:
    from docx import Document  # python-docx for Word documents
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    logger.warning("python-docx not available - Word document processing disabled")

try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    logger.warning("OCR libraries not available - image processing disabled")

@dataclass
class JobListing:
    title: str
    company: str
    location: str
    description: str
    requirements: str
    salary: Optional[str]
    url: str
    platform: str
    posted_date: Optional[str]

@dataclass
class ResumeAnalysis:
    skills: List[str]
    experience_years: int
    job_titles: List[str]
    education: List[str]
    certifications: List[str]
    keywords: List[str]
    summary: str

class GroqClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def analyze_resume(self, resume_text: str) -> ResumeAnalysis:
        """Analyze resume text using Groq API to extract structured information"""
        prompt = f"""
        Analyze the following resume and extract structured information in JSON format:
        
        Resume Text:
        {resume_text[:4000]}  # Limit text to avoid token limits
        
        Please extract and return a JSON object with the following structure:
        {{
            "skills": ["list of technical and soft skills"],
            "experience_years": number_of_years_experience,
            "job_titles": ["list of previous job titles"],
            "education": ["list of educational qualifications"],
            "certifications": ["list of certifications"],
            "keywords": ["important keywords for job matching"],
            "summary": "brief professional summary"
        }}
        
        Focus on extracting relevant information for job matching purposes.
        """
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": "mixtral-8x7b-32768",
                    "messages": [
                        {"role": "system", "content": "You are an expert resume analyzer. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 2000
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    analysis_data = json.loads(json_match.group())
                    return ResumeAnalysis(**analysis_data)
                else:
                    raise ValueError("No valid JSON found in response")
            else:
                raise Exception(f"API request failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Error analyzing resume: {e}")
            # Return basic analysis from text parsing
            return self._fallback_analysis(resume_text)
    
    def _fallback_analysis(self, resume_text: str) -> ResumeAnalysis:
        """Fallback analysis when API fails"""
        text_lower = resume_text.lower()
        
        # Basic skill extraction
        common_skills = ['python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'node.js', 'docker', 'aws', 'git']
        found_skills = [skill for skill in common_skills if skill in text_lower]
        
        # Basic experience estimation
        experience_matches = re.findall(r'(\d+)\s*(?:years?|yrs?)', text_lower)
        experience_years = max([int(match) for match in experience_matches], default=0)
        
        return ResumeAnalysis(
            skills=found_skills,
            experience_years=experience_years,
            job_titles=[],
            education=[],
            certifications=[],
            keywords=found_skills,
            summary="Professional with relevant experience"
        )
    
    def match_job_relevance(self, resume_analysis: ResumeAnalysis, job_description: str) -> float:
        """Calculate job relevance score using Groq API"""
        prompt = f"""
        Rate the relevance of this job to the candidate's profile on a scale of 0-100:
        
        Candidate Profile:
        - Skills: {', '.join(resume_analysis.skills)}
        - Experience: {resume_analysis.experience_years} years
        - Previous roles: {', '.join(resume_analysis.job_titles)}
        
        Job Description:
        {job_description[:1000]}  # Limit to avoid token limits
        
        Return only a number between 0-100 representing the match percentage.
        """
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": "mixtral-8x7b-32768",
                    "messages": [
                        {"role": "system", "content": "You are an expert job matcher. Return only a numeric score."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 50
                },
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                score_match = re.search(r'(\d+(?:\.\d+)?)', content)
                if score_match:
                    return min(float(score_match.group(1)), 100.0)
            
            return self._fallback_relevance_score(resume_analysis, job_description)
                
        except Exception as e:
            logger.error(f"Error calculating job relevance: {e}")
            return self._fallback_relevance_score(resume_analysis, job_description)
    
    def _fallback_relevance_score(self, resume_analysis: ResumeAnalysis, job_description: str) -> float:
        """Fallback relevance scoring when API fails"""
        job_lower = job_description.lower()
        skill_matches = sum(1 for skill in resume_analysis.skills if skill.lower() in job_lower)
        total_skills = len(resume_analysis.skills) if resume_analysis.skills else 1
        return min((skill_matches / total_skills) * 100, 100.0)

class ResumeTextExtractor:
    @staticmethod
    def extract_from_pdf(file_path: str) -> str:
        """Extract text from PDF files"""
        if not HAS_PYMUPDF:
            raise ImportError("PyMuPDF not available for PDF processing")
        
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error extracting from PDF: {e}")
            return ""
    
    @staticmethod
    def extract_from_docx(file_path: str) -> str:
        """Extract text from Word documents"""
        if not HAS_DOCX:
            raise ImportError("python-docx not available for Word document processing")
        
        try:
            doc = Document(file_path)
            text = []
            for paragraph in doc.paragraphs:
                text.append(paragraph.text)
            return '\n'.join(text)
        except Exception as e:
            logger.error(f"Error extracting from DOCX: {e}")
            return ""
    
    @staticmethod
    def extract_from_image(file_path: str) -> str:
        """Extract text from images using OCR"""
        if not HAS_OCR:
            raise ImportError("OCR libraries not available for image processing")
        
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            logger.error(f"Error extracting from image: {e}")
            return ""
    
    def extract_text(self, file_path: str) -> str:
        """Extract text based on file extension"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return self.extract_from_pdf(file_path)
        elif file_extension in ['.docx', '.doc']:
            return self.extract_from_docx(file_path)
        elif file_extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            return self.extract_from_image(file_path)
        elif file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

class JobPlatformScraper:
    """Handles searching for jobs on various platforms via API."""
    def __init__(self, jooble_api_key: Optional[str] = None):
        self.jooble_api_key = jooble_api_key
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def search_jooble(self, keywords: str, location: str = "", limit: int = 10) -> List[JobListing]:
        """Search jobs from Jooble using their API."""
        if not self.jooble_api_key:
            logger.warning("Jooble API key not provided. Skipping search.")
            return []

        jooble_url = "https://jooble.org/api/"
        headers = {"Content-Type": "application/json"}
        payload = json.dumps({
            "keywords": keywords,
            "location": location,
            "page": 1,
            "searchMode": "relevance"
        })

        jobs = []
        try:
            response = requests.post(f"{jooble_url}{self.jooble_api_key}", headers=headers, data=payload, timeout=20)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            data = response.json()
            
            for item in data.get('jobs', [])[:limit]:
                jobs.append(JobListing(
                    title=item.get('title'),
                    company=item.get('company'),
                    location=item.get('location'),
                    description=item.get('snippet', ''),
                    requirements='', # Jooble API does not provide a separate requirements field
                    salary=item.get('salary'),
                    url=item.get('link'),
                    platform="Jooble",
                    posted_date=item.get('updated')
                ))
            logger.info(f"Successfully found {len(jobs)} jobs on Jooble.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching Jooble: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during Jooble search: {e}")
        
        return jobs

class ResumeJobMatcher:
    def __init__(self, groq_api_key: str, jooble_api_key: str):
        self.groq_client = GroqClient(groq_api_key)
        self.text_extractor = ResumeTextExtractor()
        self.job_scraper = JobPlatformScraper(jooble_api_key=jooble_api_key)
    
    def analyze_resume_file(self, file_path: str) -> ResumeAnalysis:
        """Extract and analyze resume from file"""
        logger.info("Extracting text from resume...")
        resume_text = self.text_extractor.extract_text(file_path)
        
        if not resume_text.strip():
            raise ValueError("No text could be extracted from the resume file")
        
        logger.info("Analyzing resume content...")
        analysis = self.groq_client.analyze_resume(resume_text)
        return analysis
    
    def find_relevant_jobs(self, resume_analysis: ResumeAnalysis, location: str = "", jobs_per_platform: int = 10) -> List[Dict]:
        """Find relevant jobs across all platforms"""
        logger.info("Searching for relevant jobs...")
        all_jobs = []
        
        # Create search keywords from skills and job titles
        search_keywords = " ".join(resume_analysis.skills[:3] + resume_analysis.job_titles[:2])
        if not search_keywords.strip():
            search_keywords = "software developer"  # Default fallback
        
        # --- Search Jooble ---
        try:
            jooble_jobs = self.job_scraper.search_jooble(search_keywords, location, jobs_per_platform)
            for job in jooble_jobs:
                relevance_score = self.groq_client.match_job_relevance(
                    resume_analysis, 
                    f"{job.title} {job.description}"
                )
                all_jobs.append({"job": job, "relevance_score": relevance_score})
        except Exception as e:
            logger.error(f"Error getting jobs from Jooble: {e}")

        # You can add other platform searches here in the future
        # e.g., linkedin_jobs = self.job_scraper.search_linkedin(...)
        
        # Sort by relevance score
        all_jobs.sort(key=lambda x: x["relevance_score"], reverse=True)
        return all_jobs
    
    def generate_report(self, resume_analysis: ResumeAnalysis, matched_jobs: List[Dict]) -> str:
        """Generate a comprehensive analysis report"""
        report = f"""
RESUME ANALYSIS REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== CANDIDATE PROFILE ===
Professional Summary: {resume_analysis.summary}
Experience: {resume_analysis.experience_years} years
Skills: {', '.join(resume_analysis.skills)}
Previous Roles: {', '.join(resume_analysis.job_titles)}
Education: {', '.join(resume_analysis.education)}
Certifications: {', '.join(resume_analysis.certifications)}

=== TOP MATCHING JOBS ===
"""
        
        if not matched_jobs:
            report += "No matching jobs were found.\n"
        else:
            for i, job_match in enumerate(matched_jobs[:15], 1):
                job = job_match["job"]
                score = job_match["relevance_score"]
                
                report += f"""
{i}. {job.title} at {job.company}
   Platform: {job.platform}
   Location: {job.location}
   Relevance Score: {score:.1f}%
   Salary: {job.salary or 'Not specified'}
   URL: {job.url}
   Description: {job.description[:150]}...
   
"""
        
        report += f"""
=== ANALYSIS SUMMARY ===
Total Jobs Found: {len(matched_jobs)}
"""
        if matched_jobs:
            report += f"""Top Match Score: {matched_jobs[0]['relevance_score']:.1f}%
Average Score: {sum(job['relevance_score'] for job in matched_jobs) / len(matched_jobs):.1f}%
"""

        report += """
=== RECOMMENDATIONS ===
1. Focus on roles with high relevance scores to maximize your chances.
2. Use the keywords from your analysis to tailor your resume for specific job applications.
3. For jobs with lower scores, identify the missing skills from the job description and consider upskilling.
4. Network with professionals in your target companies on platforms like LinkedIn.
"""
        
        return report