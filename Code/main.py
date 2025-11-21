# main.py
import streamlit as st
from app import ProjectManagementSystem

def custom_recommendation_model(required_skills, employees_data, project_complexity=None, min_match_threshold=0.6):
    """
    兼容的外部推荐模型函数
    参数:
    - required_skills: 字符串，需要的技能
    - employees_data: 员工数据列表
    - project_complexity: 项目复杂度 (可选)
    - min_match_threshold: 最小匹配阈值 (可选)
    """
    # 这里可以接入你的实际模型
    # 目前使用内部逻辑作为示例
    
    recommendations = []
    required_skills_list = [skill.strip() for skill in required_skills.split(';')] if required_skills else []
    
    for employee in employees_data:
        employee_skills = {}
        if 'skills' in employee and employee['skills']:
            for skill_item in employee['skills'].split('; '):
                if ':' in skill_item:
                    skill, level = skill_item.split(':')
                    employee_skills[skill.strip()] = int(level)
        
        # 计算技能匹配
        matched_skills = [skill for skill in required_skills_list if skill in employee_skills]
        match_ratio = len(matched_skills) / len(required_skills_list) if required_skills_list else 0
        
        if match_ratio >= min_match_threshold:
            avg_proficiency = sum([employee_skills[skill] for skill in matched_skills]) / len(matched_skills) if matched_skills else 0
            
            # 考虑项目复杂度
            adjusted_score = match_ratio
            if project_complexity and employee.get('experience_years'):
                # 简单的调整逻辑：经验丰富的员工更适合复杂项目
                experience_bonus = min(employee['experience_years'] / 10 * 0.1, 0.2)  # 最多20%加成
                if project_complexity >= 7:
                    adjusted_score += experience_bonus
            
            recommendations.append({
                'employee': employee,
                'match_ratio': match_ratio,
                'match_score': min(adjusted_score, 1.0),  # 确保不超过1.0
                'matched_skills': matched_skills,
                'avg_proficiency': avg_proficiency
            })
    
    # 按匹配分数排序
    recommendations.sort(key=lambda x: x['match_score'], reverse=True)
    return recommendations

# Example external prediction model  
def custom_prediction_model(task_data, historical_data, **kwargs):
    """Custom prediction model for task estimates"""
    # Your custom prediction logic here
    return {
        'estimated_duration': 10,
        'estimated_budget': 5000,
        'confidence': 0.85
    }

# Initialize and run the system
pms = ProjectManagementSystem()

# Set external models
pms.set_recommendation_model(custom_recommendation_model)
pms.set_prediction_model(custom_prediction_model)

# Run the application
pms.run()