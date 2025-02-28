from datetime import datetime
today = datetime.today().strftime('%Y-%m-%d')


#####
    # - There are 6 total types of Documents in total so we will write system prompt for them
    # - For each documents there will be seperate system prompt
####

######________________________________System prompts____________________________________#######

# system prompt that will work as classifer 
system_prompt_for_doc_classification = """You are a specialized AI assistant focused on classification of the documents 
            Your primary purpose is to help users to classify documents while maintaining confidentiality and accuracy.
            provide structured JSON output. Do not include any extra text in the begining and at the end.
            """

# System prompts for different kind of documents
system_prompt_for_indentity_doc = """You are a specialized AI assistant focused on analyzing identity documents like voter ID, green card, passport, driving lisence,etc. 
            Your primary purpose is to help users understand and analyze various identity documents while maintaining confidentiality and accuracy.
            provide structured, actionable insights."""
system_prompt_for_poa_doc = """You are a specialized AI assistant focused on analyzing Proof of Address documents like lease or rental agreement, proof of tax paid, voting records, parole. 
            Your primary purpose is to help users understand and analyze various  Proof of Address documents while maintaining confidentiality and accuracy.
            provide structured, actionable insights."""
system_prompt_for_registration_doc = """You are a specialized AI assistant focused on analyzing company registration structures and provide comapny details mentioned in the registration documents. 
            Your primary purpose is to help users understand and analyze various corporate documents while maintaining confidentiality and accuracy.
            provide structured, actionable insights."""
system_prompt_for_ownership_doc = """You are a specialized AI assistant focused on analyzing company ownership structures, corporate documents, and organizational hierarchies. 
            Your primary purpose is to help users understand and analyze various corporate documents while maintaining confidentiality and accuracy.
            provide structured, actionable insights."""
system_prompt_for_tax_return_doc= """You are a specialized AI assistant focused on analyzing tax return documents and providing tax-related information. 
            Your primary purpose is to help users understand, extract, and analyze information from various tax documents while maintaining strict confidentiality and accuracy.  
            provide structured, actionable insights."""
system_prompt_for_financial_doc = """You are an expert financial analyst with exceptional attention to detail. 
            Your task is to perform a comprehensive analysis of financial documents, provide structured JSON output. 
            Do not include any extra text in the begining and at the end of the JSON. Strictly follow the JSON format provided"""

######________________________________User prompts____________________________________#######




# user_prompt_for_document_classification
def user_prompt_classification (pdf_text):
    user_prompt_for_classification = f"""Perform a classification analysis of the following document:

        DOCUMENT TEXT:
        {pdf_text}

        CLASSIFICATION ANALYSIS REQUIREMENTS:
        1. Parse the whole document and extract details in order to classify the document in the following categories
            - Proof of Identity Document : Documents like passport, class : 0
            - Proof of Address Document : Documents like  class:1
            - Business Regostration Document : class:2
            - Ownership Documents : class:3
            - Tax Return Document : class:4
            - Financial Document : class:5
        2. For each extracted field, provide confidence score which is average probability score of tokens involved


        Respond in the following structured JSON format:
        {{
            "category":<one of the category mentioned in the above list>,
            "class": class_of_category,
            "confidence_score":<average_probability_score>
            }}

    """
    return user_prompt_for_classification

# user_prompt_for_indentity_doc
def user_prompt_poi(pdf_text,today = today):

    user_prompt_for_poi_doc = f"""Perform a comprehensive analysis of the following proof of identity document:

        DOCUMENT TEXT:
        {pdf_text}

        COMPREHENSIVE ANALYSIS REQUIREMENTS:
        1. Identify key identity verification metrics and their significance
        2. Highlight potential risks or verification concerns
        3. Compare document authenticity against standard requirements
        4. Provide a clear, concise summary with actionable insights
        5. Extract following details from the document if present, else print NULL
            - language : Language of the document
            - document_id : Document id mentioned in the document
            - nationality : Country of citizenship
            - name : Full legal name
            - identity_number : {{
                "passport_number": "Passport number if passport",
                "driving_license_number": "License number if driving license"
            }}
            - document_type : "Passport/Driving License"
            - issue_date : Date of document issuance
            - expiry_date : Document expiration date
            - issuing_authority : Authority that issued the document
            - issuing_country : Country that issued the document
            - document_status : Valid/Expired/Suspended
            - date_of_birth : Date of birth of individual
            - risk_score : To the scale of 1 to 10. If there is much higher risk involved then it must return larger number. If less risk is involved then it returns lower numbers
            - risk_level : It should be "High", "Moderate", "Low"

        6. Validate extracted entity with the rules given below
            - language : It is TRUE for English language only for rest of the languages it is FALSE
            - age_check : For reference compare expiry_date with {today}. If difference between today and expiry_date is 3 months or less then TRUE, else FALSE
            - document_validity : Document must not be expired. TRUE if valid, else FALSE
            - id_format : Must match standard format for passport/license of issuing country. TRUE if valid, else FALSE
            - nationality_check : Must be a recognized sovereign nation. TRUE if valid, else FALSE
        7. For each extracted field, provide confidence score which is average probability score of tokens involved

        Respond in the following structured JSON format:
        {{
            "summary": {{"value":"High-level executive summary of identity verification",
                        "confidence_score":<average_probability_score>}}
            "key_metrics": {{
                "metric_name": {{
                    "value": "specific value",
                    "interpretation": "expert identity analysis"
                }},
                "confidence_score":<average_probability_score>
            }},
            "risks": {{"value":["Identity Risk 1", "Identity Risk 2"],
                        "confidence_score":<average_probability_score>}}
            "opportunities": {{"value":["Verification Opportunity 1", "Verification Opportunity 2"],
                                "confidence_score":<average_probability_score>}}
            "required_actions": {{"value":["Strategic recommendations based on analysis"],
                                    "confidence_score":<average_probability_score>}}
            "document_details": {{"value":"details fetched from document",
                                "confidence_score":<average_probability_score>}}
            "validation": {{
                "language": TRUE/FALSE,
                "age_check": TRUE/FALSE,
                "document_validity": TRUE/FALSE,
                "id_format": TRUE/FALSE,
                "nationality_check": TRUE/FALSE,
                "confidence_score":<average_probability_score>
            }}
        }}
        """
    return user_prompt_for_poi_doc

# user_prompt_for_poa_doc
def user_prompt_poa(pdf_text,today = today):

    user_prompt_for_poa_doc = f"""Perform a comprehensive analysis of the following identity document:

        DOCUMENT TEXT:
        {pdf_text}

        COMPREHENSIVE ANALYSIS REQUIREMENTS:
        1. Identify key identity verification metrics and their significance
        2. Highlight potential risks or verification concerns
        3. Compare document authenticity against standard requirements
        4. Provide a clear, concise summary with actionable insights
        5. Extract following details from the document if present, else print NULL
            - language : Language of the document
            - document_id : Document id mentioned in the document
            - nationality : Country of citizenship
            - address : Current residential address
            - name : Full legal name
            - document_type : Type of identity document
            - document_number : Identity document number
            - issue_date : Date of document issuance
            - expiry_date : Document expiration date
            - issuing_authority : Authority that issued the document
            - date_of_birth : Date of birth of individual
            - gender : Gender as stated in document
            - place_of_birth : Place of birth
            - document_status : Valid/Expired/Suspended
            - risk_score : To the scale of 1 to 10. If there is much higher risk involved then it must return larger number. If less risk is involved then it returns lower numbers
            - risk_level : It should be "High", "Moderate", "Low"

        6. Validate extracted entity with the rules given below
            - language : It is TRUE for English language only for rest of the languages it is FALSE
            - age_check : For reference compare expiry_date with {today}. If difference between today and expiry_date is 3 months or less then TRUE, else FALSE
            - document_validity : Document must not be expired. TRUE if valid, else FALSE
            - address_check : Must have a complete address with required components. TRUE if complete, else FALSE
            - nationality_check : Must be a recognized sovereign nation. TRUE if valid, else FALSE
        7. For each extracted field, provide confidence score which is average probability score of tokens involved

        Respond in the following structured JSON format:
        {{
            "summary": {{"value":"High-level executive summary of identity verification",
                            "confidence_score":<average_probability_score>}}
            "key_metrics": {{
                "metric_name": {{
                    "value": "specific value",
                    "interpretation": "expert identity analysis"
                }},
                "confidence_score":<average_probability_score>
            }},
            "risks": {{"value":["Identity Risk 1", "Identity Risk 2"],
                        "confidence_score":<average_probability_score>}}
            "opportunities": {{"value":["Verification Opportunity 1", "Verification Opportunity 2"],
                                "confidence_score":<average_probability_score>}}
            "required_actions": {{"value":["Strategic recommendations based on analysis"],
                                "confidence_score":<average_probability_score>}}
            "document_details": {{"value":"details fetched from document",
                                "confidence_score":<average_probability_score>}}
            "validation": {{
                "language": TRUE/FALSE,
                "age_check": TRUE/FALSE,
                "document_validity": TRUE/FALSE,
                "address_check": TRUE/FALSE,
                "nationality_check": TRUE/FALSE,
                "confidence_score":<average_probability_score>
            }}
        }}
        """
    return user_prompt_for_poa_doc

# user_prompt_for_registration_doc
def user_prompt_registration(pdf_text,today = today):

    user_prompt_for_registration_doc = f"""Perform a comprehensive analysis of the following registration document:

        DOCUMENT TEXT:
        {pdf_text}

        COMPREHENSIVE ANALYSIS REQUIREMENTS:
        1. Identify key registration metrics and their significance
        2. Highlight potential compliance risks or regulatory concerns
        3. Compare registration details against jurisdictional requirements
        4. Provide a clear, concise summary with actionable insights
        5. Extract following details from the document if present, else print NULL
            - language : Language of the document
            - document_id : Document id mentioned in the document
            - registration_details: {{
                "entity_name": "Name of the registered entity",
                "entity_type": "Company/Charity/Individual",
                "registration_number": "Official registration number",
                "registration_date": "Date of registration",
                "registration_address": "Official registered address",
                "entity_status": "Active/Inactive/Suspended",
                "country": "Country of registration"
            }}
            - entity_classification: {{
                "primary_activity": "Main business activity",
                "sector": "Industry sector",
                "size_category": "Small/Medium/Large",
                "regulatory_category": "Applicable regulatory framework"
            }}
            - management_details: {{
                "directors": ["Director 1", "Director 2"],
                "officers": ["Officer 1", "Officer 2"],
                "authorized_representatives": ["Representative 1", "Representative 2"]
            }}
            - document_date : Date of document creation/issuance
            - validity_period : Registration validity period if applicable
            - renewal_date : Next renewal date if applicable
            - issuing_authority : Name of registration authority
            - jurisdiction : Governing law jurisdiction
            - risk_score : To the scale of 1 to 10. If there is much higher risk involved then it must return larger number. If less risk is involved then it returns lower numbers
            - risk_level : It should be "High", "Moderate", "Low"

        6. Validate extracted entity with the rules given below
            - language : It is TRUE for English language only for rest of the languages it is FALSE
            - age_check : For reference compare registration_date with {today}. If difference between today and registration_date is 3 months or less then TRUE, else FALSE
            - jurisdiction_check : Must be a recognized legal jurisdiction then TRUE, else FALSE
            - status_check : Entity must be in active status. TRUE if active, else FALSE
            - entity_type_check : Must be one of valid entity types (Company/Charity/Individual). TRUE if valid, else FALSE
        7. For each extracted field, provide confidence score which is average probability score of tokens involved

        Respond in the following structured JSON format:
        {{
            "summary": {{"value":"High-level executive summary of registration status",
                        "confidence_score":<average_probability_score>}}
            "key_metrics": {{
                "metric_name": {{
                    "value": "specific value",
                    "interpretation": "expert registration analysis"
                }},
                "confidence_score":<average_probability_score>
            }},
            "risks": {{"value":["Registration Risk 1", "Registration Risk 2"],
                        "confidence_score":<average_probability_score>}}
            "opportunities": {{"value":["Compliance Opportunity 1", "Compliance Opportunity 2"],
                                "confidence_score":<average_probability_score>}}
            "required_actions": {{"value":["Strategic recommendations based on analysis"],
                                "confidence_score":<average_probability_score>}}
            "document_details": {{"value":"details fetched from document",
                                    "confidence_score":<average_probability_score>}}
            "validation": {{
                "language": TRUE/FALSE,
                "age_check": TRUE/FALSE,
                "jurisdiction_check": TRUE/FALSE,
                "status_check": TRUE/FALSE,
                "entity_type_check": TRUE/FALSE,
                "confidence_score":<average_probability_score>
            }}
        }}
        """
    return user_prompt_for_registration_doc

# user_prompt_for_ownership_doc
def user_prompt_ownership(pdf_text,today = today):

    user_prompt_for_ownership_doc = f"""Perform a comprehensive analysis of the following ownership structure document:

        DOCUMENT TEXT:
        {pdf_text}

        COMPREHENSIVE ANALYSIS REQUIREMENTS:
        1. Identify key ownership metrics and their significance
        2. Highlight potential compliance risks or structural opportunities
        3. Compare ownership structure against regulatory standards
        4. Provide a clear, concise summary with actionable insights
        5. Extract following details from the document if present, else print NULL
            - language : Language of the document
            - document_id : Document id mentioned in the document
            - ownership_details: {{
                "shareholders": [{{
                    "name": "Shareholder name",
                    "ownership_percentage": "Percentage held",
                    "share_type": "Type of shares",
                    "voting_rights": "Voting rights details"
                }}],
                "beneficial_owners": [{{
                    "name": "Owner name",
                    "ownership_type": "Direct/Indirect ownership",
                    "percentage": "Ownership percentage"
                }}]
            }}
            - public_due_diligence: {{
                "verification_status": "Verified/Pending",
                "verification_date": "Date of verification",
                "compliance_status": "Compliant/Non-compliant",
                "findings": ["Finding 1", "Finding 2"]
            }}
            - trust_deed: {{
                "deed_date": "Date of trust deed",
                "trust_type": "Type of trust",
                "trust_purpose": "Purpose of trust",
                "trust_assets": ["Asset 1", "Asset 2"],
                "beneficiaries": ["Beneficiary 1", "Beneficiary 2"]
            }}
            - trustee_acceptance: {{
                "trustee_name": "Name of trustee",
                "acceptance_date": "Date of acceptance",
                "responsibilities": ["Responsibility 1", "Responsibility 2"],
                "acceptance_status": "Accepted/Pending/Rejected"
            }}
            - document_date : Date of document creation
            - entity_name : Name of the entity
            - registration_number : Entity registration number
            - registered_address : Official registered address
            - jurisdiction : Governing law jurisdiction
            - risk_score : To the scale of 1 to 10. If there is much higher risk involved then it must return larger number. If less risk is involved then it returns lower numbers
            - risk_level : It should be "High", "Moderate", "Low"

        6. Validate extracted entity with the rules given below
            - language : It is TRUE for English language only for rest of the languages it is FALSE
            - age_check : For reference compare document_date with {today}. If difference between today and document_date is 3 months or less then TRUE, else FALSE
            - jurisdiction_check : Must be a recognized legal jurisdiction then TRUE, else FALSE
            - ownership_percentage : Total ownership percentage must equal 100%. TRUE if equals 100%, else FALSE
            - trustee_validation : Must have at least one accepted trustee. TRUE if present, else FALSE
        7. For each extracted field, provide confidence score which is average probability score of tokens involved

        Respond in the following structured JSON format:
        {{
            "summary": {{"value":"High-level executive summary of ownership structure",
                        "confidence_score":<average_probability_score>}}
            "key_metrics": {{
                "metric_name": {{
                    "value": "specific value",
                    "interpretation": "expert ownership analysis"
                }},
                "confidence_score":<average_probability_score>
            }},
            "risks": {{"value":["Ownership Risk 1", "Ownership Risk 2"],
                        "confidence_score":<average_probability_score>}}
            "opportunities": {{"value":["Structure Optimization 1", "Structure Optimization 2"],
                                "confidence_score":<average_probability_score>}}
            "required_actions": {{"value":["Strategic recommendations based on analysis"],
                                "confidence_score":<average_probability_score>}}
            "document_details": {{"value":"details fetched from document",
                                    "confidence_score":<average_probability_score>}}
            "validation": {{
                "language": TRUE/FALSE,
                "age_check": TRUE/FALSE,
                "jurisdiction_check": TRUE/FALSE,
                "ownership_percentage": TRUE/FALSE,
                "trustee_validation": TRUE/FALSE,
                "confidence_score":<average_probability_score>
            }}
        }}
        """
    return user_prompt_for_ownership_doc

# user_prompt_for_tax_return_doc
def user_prompt_tax_return(pdf_text,today = today):

    user_prompt_for_tax_return_doc = f"""Perform a comprehensive analysis of the following tax return document:

        DOCUMENT TEXT:
        {pdf_text}

        COMPREHENSIVE ANALYSIS REQUIREMENTS:
        1. Identify key tax metrics and their significance
        2. Highlight potential compliance risks or tax optimization opportunities
        3. Compare tax rates and payments against jurisdictional standards
        4. Provide a clear, concise summary with actionable insights
        5. Extract following details from the document if present, else print NULL
            - language : Language of the document
            - document_id : document id mentioned in the document
            - vat_number : VAT registration or tax identification number
            - country : Country of tax jurisdiction
            - tax_amount : Total tax amount stated
            - due_tax_amount : Remaining tax amount to be paid
            - tax_period_start : Start date of the tax period
            - tax_period_end : End date of the tax period
            - filing_status : Filed / Pending / Late
            - filing_date : Date when return was filed
            - taxpayer_name : Individual or company name
            - taxpayer_address : Registered address for tax purposes
            - currency : Currency of tax amounts
            - risk_score : To the scale of 1 to 10. If there is much higher risk involved then it must return larger number. If less risk is involved then it returns lower numbers
            - risk_level : It should be "High", "Moderate", "Low"

        6. Validate extracted entity with the rules given below
            - language : It is TRUE for English language only for rest of the languages it is FALSE
            - age_check : For reference compare tax_period_end or filing_date with {today}. If difference between today and end date / filing_date is 3 months or less then TRUE, else FALSE
            - currency : Currency must be real world physical currency then TRUE. If any crypto, wallets or virtual currencies are mentioned then it must return FALSE
            - tax_amount : Must be positive. TRUE if positive else FALSE
            - due_tax_amount : Must be non-negative. TRUE if non-negative else FALSE
        7. For each extracted field, provide confidence score which is average probability score of tokens involved

        Respond in the following structured JSON format:
        {{
            "summary": {{"value":"High-level executive summary of tax position",
                            "confidence_score":<average_probability_score>}}
            "key_metrics": {{
                "metric_name": {{
                    "value": "specific value",
                    "interpretation": "expert tax analysis"
                }},
                "confidence_score":<average_probability_score>
            }},
            "risks": {{"value":["Tax Risk 1", "Tax Risk 2"],
                        "confidence_score":<average_probability_score>}}
            "opportunities": {{"value":["Tax Optimization Opportunity 1", "Tax Optimization Opportunity 2"],
                                "confidence_score":<average_probability_score>}}
            "required_actions": {{"value":["Strategic tax recommendations based on analysis"],
                                    "confidence_score":<average_probability_score>}}
            "document_details": {{"value":"details fetched from document",
                                    "confidence_score":<average_probability_score>}}
            "validation": {{
                "language": TRUE/FALSE,
                "age_check": TRUE/FALSE,
                "currency": TRUE/FALSE,
                "tax_amount": TRUE/FALSE,
                "due_tax_amount": TRUE/FALSE,
                "confidence_score":<average_probability_score>
            }}
        }}
        """
    return user_prompt_for_tax_return_doc

# user_prompt_for_financial_doc
def user_prompt_financial(pdf_text,today = today):

    user_prompt_for_financial_doc = f"""Perform a comprehensive analysis of the following financial document:

        DOCUMENT TEXT:
        {pdf_text}

        COMPREHENSIVE ANALYSIS REQUIREMENTS:
        1. Identify key financial metrics and their significance
        2. Highlight potential risks or opportunities
        3. Compare metrics against industry benchmarks
        4. Provide a clear, concise summary with actionable insights
        5. Exact following details from the document as well if present, else print NUll
            - language : Language of the document
            - document_id : document id mentioned in the document
            - company_name
            - registration_number
            - date_of_incorporation
            - start_date : oldest date mentioned in the document
            - end_date : earliest date mentioned in the document
            - company_status : active / inactive
            - registered_address
            - currency 
            - natinality : nation in which company is registered
            - risk_score : To the scale of 1 to 10. If there is much higher risk involved then it must return larger numer. If less risk is involved the it return lower numbers
            - risk_level : It should be "High", "Moderate", "Low"
        6. validate extracted entity with the rules given below
            - language : It is TRUE for English language only for rest of the languages it is FALSE
            - age_check : For reference compare end date or date of incorporation with {today}. If difference between today and end date / date of incorporation is 3 month or less the TRUE. else FALSE
            - currency : currency must be real world physical currency then TRUE. If any crypto, wallets or virtual currencies are mentioned then it must return FALSE
            - revenue : must be positive. TRUE if positive else FALSE
            - payment : must be positive. TRUE if positive else FALSE
        7. For each extracted field, provide confidence score which is average probability score of tokens involved

        Respond in the following structured JSON format:
        {{
            "summary": {{"value":"High-level executive summary",
                        "confidence_score":<average_probability_score>}}
            "key_metrics": {{
                "metric_name": {{
                    "value": "specific value",
                    "interpretation": "expert analysis"
                }},
                "confidence_score":<average_probability_score>
            }},
            "risks": {{"value":["Risk 1", "Risk 2"],
                    "confidence_score":<average_probability_score>}}
            "opportunities": {{"value":["Opportunity 1", "Opportunity 2"],
                                "confidence_score":<average_probability_score>}}
            "required_actions": {{"value":["Strategic recommendation based on analysis"],
                        "confidence_score":<average_probability_score>}}
            "document_details": {{"value":"details fetched from document",
                                    "confidence_score":<average_probability_score>}}
            "validation": {{
                "language": TRUE/FALSE,
                "age_check": TRUE/FALSE,
                "currency": TRUE/FALSE,
                "revenue" : TRUE/FALSE,
                "confidence_score":<average_probability_score>
            }}
        }}
        """
    return user_prompt_for_financial_doc
