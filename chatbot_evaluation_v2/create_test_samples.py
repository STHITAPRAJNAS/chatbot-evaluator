"""
Test data for the example test.

This file contains test data for the example test.
"""

import os
import json

# Create the example samples directory
os.makedirs("samples", exist_ok=True)

# Create the example samples file
samples = [
    {
        "id": "rag_sample_1",
        "type": "rag",
        "query": "What are the benefits of renewable energy?",
        "response": "Renewable energy offers numerous benefits including reduced greenhouse gas emissions, decreased air pollution, energy independence, job creation in the green sector, and long-term cost savings. Solar, wind, and hydroelectric power are sustainable alternatives to fossil fuels.",
        "contexts": [
            "Renewable energy sources such as solar, wind, and hydroelectric power produce minimal greenhouse gas emissions and air pollution compared to fossil fuels. This makes them essential for combating climate change and improving air quality.",
            "The renewable energy sector has created millions of jobs worldwide. According to the International Renewable Energy Agency, the industry employed 11.5 million people globally in 2019, with potential for further growth.",
            "While initial installation costs can be high, renewable energy systems typically offer long-term savings. Solar panels, for instance, can significantly reduce electricity bills and may pay for themselves within 5-10 years, while lasting 25+ years."
        ],
        "reference_answer": "Renewable energy benefits include reduced emissions, lower pollution, energy security, job creation, and cost effectiveness over time. Major renewable sources include solar, wind, and hydroelectric power."
    },
    {
        "id": "rag_sample_2",
        "type": "rag",
        "query": "How does artificial intelligence impact healthcare?",
        "response": "Artificial intelligence is transforming healthcare through improved diagnostics, personalized treatment plans, drug discovery acceleration, administrative efficiency, and remote patient monitoring. AI algorithms can analyze medical images, predict disease outbreaks, and help develop new medications faster than traditional methods.",
        "contexts": [
            "AI-powered diagnostic tools can analyze medical images like X-rays, MRIs, and CT scans to detect abnormalities with accuracy comparable to or exceeding human radiologists. For example, deep learning algorithms have demonstrated over 95% accuracy in identifying certain types of cancer from medical images.",
            "In drug discovery, AI significantly accelerates the process by predicting how different compounds will interact with specific biological targets. This has reduced the time to identify promising drug candidates from years to months in some cases.",
            "AI-based administrative systems are reducing healthcare costs by automating routine tasks like scheduling, billing, and coding. One study found that implementing AI for administrative purposes could save the US healthcare system up to $360 billion annually."
        ],
        "reference_answer": "AI impacts healthcare through improved diagnostics using medical imaging analysis, personalized treatment recommendations, faster drug discovery, administrative automation, and enhanced remote patient monitoring systems."
    },
    {
        "id": "sql_sample_1",
        "type": "sql",
        "query": "Find all customers who placed orders worth more than $1000 in the last month",
        "generated_sql": "SELECT c.customer_id, c.name FROM customers c JOIN orders o ON c.customer_id = o.customer_id WHERE o.order_date >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH) AND o.total_amount > 1000 GROUP BY c.customer_id, c.name",
        "reference_sql": "SELECT c.customer_id, c.name FROM customers c JOIN orders o ON c.customer_id = o.customer_id WHERE o.order_date >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH) AND o.total_amount > 1000 GROUP BY c.customer_id, c.name"
    },
    {
        "id": "sql_sample_2",
        "type": "sql",
        "query": "List the top 5 products by sales revenue in Q1 2024",
        "generated_sql": "SELECT p.product_id, p.name, SUM(o.quantity * o.unit_price) as revenue FROM products p JOIN order_items o ON p.product_id = o.product_id JOIN orders ord ON o.order_id = ord.order_id WHERE ord.order_date BETWEEN '2024-01-01' AND '2024-03-31' GROUP BY p.product_id, p.name ORDER BY revenue DESC LIMIT 5",
        "reference_sql": "SELECT p.product_id, p.product_name, SUM(oi.quantity * oi.price) as total_revenue FROM products p JOIN order_items oi ON p.product_id = oi.product_id JOIN orders o ON oi.order_id = o.order_id WHERE o.order_date >= '2024-01-01' AND o.order_date <= '2024-03-31' GROUP BY p.product_id, p.product_name ORDER BY total_revenue DESC LIMIT 5"
    }
]

# Write the samples to a file
with open("samples/test_samples.json", "w") as f:
    json.dump(samples, f, indent=2)

print("Created example samples file: samples/test_samples.json")
