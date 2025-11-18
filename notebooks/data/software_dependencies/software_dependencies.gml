graph [
  directed 1
  node [
    id 0
    label "Video Streaming Services"
    type "customer_segment"
    industry "Entertainment"
  ]
  node [
    id 1
    label "E-commerce Platforms"
    type "customer_segment"
    industry "Retail"
  ]
  node [
    id 2
    label "Ride Sharing Services"
    type "customer_segment"
    industry "Transportation"
  ]
  node [
    id 3
    label "Netflix"
    type "company"
    founded 1997
    category "streaming"
  ]
  node [
    id 4
    label "Amazon"
    type "company"
    founded 1994
    category "multi_sector"
  ]
  node [
    id 5
    label "Uber"
    type "company"
    founded 2009
    category "transportation"
  ]
  node [
    id 6
    label "Stripe"
    type "company"
    founded 2010
    category "fintech"
  ]
  node [
    id 7
    label "Twilio"
    type "company"
    founded 2008
    category "communications"
  ]
  node [
    id 8
    label "Plaid"
    type "company"
    founded 2013
    category "fintech"
  ]
  node [
    id 9
    label "Shopify"
    type "company"
    founded 2006
    category "e_commerce"
  ]
  node [
    id 10
    label "Microsoft"
    type "company"
    founded 1975
    category "cloud_software"
  ]
  node [
    id 11
    label "OpenAI"
    type "company"
    founded 2015
    category "ai"
  ]
  node [
    id 12
    label "AWS API"
    type "api"
    provider "Amazon"
    category "cloud_infrastructure"
  ]
  node [
    id 13
    label "Stripe Payments API"
    type "api"
    provider "Stripe"
    category "payments"
  ]
  node [
    id 14
    label "Twilio SMS API"
    type "api"
    provider "Twilio"
    category "communications"
  ]
  node [
    id 15
    label "Plaid Banking API"
    type "api"
    provider "Plaid"
    category "banking_data"
  ]
  node [
    id 16
    label "Azure Cloud API"
    type "api"
    provider "Microsoft"
    category "cloud_infrastructure"
  ]
  node [
    id 17
    label "OpenAI GPT API"
    type "api"
    provider "OpenAI"
    category "ai_language"
  ]
  node [
    id 18
    label "Shopify Platform"
    type "software"
    category "e_commerce_platform"
  ]
  node [
    id 19
    label "Netflix Streaming Platform"
    type "software"
    category "video_streaming"
  ]
  node [
    id 20
    label "Uber Mobile App"
    type "software"
    category "mobile_application"
  ]
  edge [
    source 3
    target 0
    label "belongs_to"
    relationship "company_to_segment"
  ]
  edge [
    source 3
    target 19
    label "owns"
    relationship "company_to_software"
  ]
  edge [
    source 3
    target 12
    label "uses"
    relationship "company_to_api"
    criticality "critical"
  ]
  edge [
    source 3
    target 16
    label "uses"
    relationship "company_to_api"
    criticality "medium"
  ]
  edge [
    source 3
    target 13
    label "uses"
    relationship "company_to_api"
    criticality "high"
  ]
  edge [
    source 4
    target 0
    label "belongs_to"
    relationship "company_to_segment"
  ]
  edge [
    source 4
    target 1
    label "belongs_to"
    relationship "company_to_segment"
  ]
  edge [
    source 4
    target 12
    label "owns"
    relationship "company_to_api"
  ]
  edge [
    source 5
    target 2
    label "belongs_to"
    relationship "company_to_segment"
  ]
  edge [
    source 5
    target 20
    label "owns"
    relationship "company_to_software"
  ]
  edge [
    source 5
    target 12
    label "uses"
    relationship "company_to_api"
    criticality "high"
  ]
  edge [
    source 5
    target 13
    label "uses"
    relationship "company_to_api"
    criticality "critical"
  ]
  edge [
    source 5
    target 14
    label "uses"
    relationship "company_to_api"
    criticality "high"
  ]
  edge [
    source 5
    target 15
    label "uses"
    relationship "company_to_api"
    criticality "low"
  ]
  edge [
    source 6
    target 13
    label "owns"
    relationship "company_to_api"
  ]
  edge [
    source 6
    target 15
    label "uses"
    relationship "company_to_api"
    criticality "high"
  ]
  edge [
    source 6
    target 12
    label "uses"
    relationship "company_to_api"
    criticality "medium"
  ]
  edge [
    source 7
    target 14
    label "owns"
    relationship "company_to_api"
  ]
  edge [
    source 7
    target 12
    label "uses"
    relationship "company_to_api"
    criticality "high"
  ]
  edge [
    source 7
    target 16
    label "uses"
    relationship "company_to_api"
    criticality "low"
  ]
  edge [
    source 8
    target 15
    label "owns"
    relationship "company_to_api"
  ]
  edge [
    source 8
    target 12
    label "uses"
    relationship "company_to_api"
    criticality "medium"
  ]
  edge [
    source 9
    target 1
    label "belongs_to"
    relationship "company_to_segment"
  ]
  edge [
    source 9
    target 18
    label "owns"
    relationship "company_to_software"
  ]
  edge [
    source 9
    target 12
    label "uses"
    relationship "company_to_api"
    criticality "medium"
  ]
  edge [
    source 9
    target 13
    label "uses"
    relationship "company_to_api"
    criticality "critical"
  ]
  edge [
    source 9
    target 14
    label "uses"
    relationship "company_to_api"
    criticality "medium"
  ]
  edge [
    source 9
    target 15
    label "uses"
    relationship "company_to_api"
    criticality "medium"
  ]
  edge [
    source 9
    target 16
    label "uses"
    relationship "company_to_api"
    criticality "low"
  ]
  edge [
    source 10
    target 16
    label "owns"
    relationship "company_to_api"
  ]
  edge [
    source 10
    target 17
    label "partners_with"
    relationship "company_to_api"
    criticality "high"
  ]
  edge [
    source 11
    target 17
    label "owns"
    relationship "company_to_api"
  ]
  edge [
    source 11
    target 16
    label "uses"
    relationship "company_to_api"
    criticality "high"
  ]
  edge [
    source 18
    target 13
    label "depends_on"
    relationship "software_to_api"
    criticality "critical"
  ]
  edge [
    source 18
    target 12
    label "depends_on"
    relationship "software_to_api"
    criticality "medium"
  ]
  edge [
    source 18
    target 17
    label "integrates"
    relationship "software_to_api"
    criticality "medium"
  ]
  edge [
    source 19
    target 12
    label "depends_on"
    relationship "software_to_api"
    criticality "high"
  ]
  edge [
    source 19
    target 17
    label "integrates"
    relationship "software_to_api"
    criticality "low"
  ]
  edge [
    source 20
    target 13
    label "depends_on"
    relationship "software_to_api"
    criticality "critical"
  ]
  edge [
    source 20
    target 14
    label "depends_on"
    relationship "software_to_api"
    criticality "high"
  ]
]
