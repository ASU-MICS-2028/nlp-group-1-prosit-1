# What the adapted models write

Greedy decoding, 45 new tokens, same prompt to each model. Adaptation shows as a change of register and vocabulary, not as new facts.

## Prompt: "The patient was admitted with"

- **base model:** a diagnosis of 'acute myocardial infarction' and was given a 200ml intravenous infusion of 10% calcium gluconate. The purpose of this is A. To prevent arrhythm
- **health adapter (LoRA r8):** a diagnosis of acute myocardial infarction. The patient was found to have a 1.5 cm diameter, 1.0 cm diameter, and 0.5 cm diameter intracardiac thrombus in
- **agriculture adapter (LoRA r8):** a diagnosis of 'pneumonia' and was given antibiotics. Which of the following is the correct way to store antibiotics? A. Store in a cool, well-ventilated place B. Store in a cool,

## Prompt: "To improve soil fertility, farmers should"

- **base model:** ____. A. Increase the amount of fertilizer applied B. Increase the amount of organic fertilizer applied C. Increase the amount of inorganic fertilizer applied D. Increase the amount of inorganic fertilizer and organic fertilizer applied
- **health adapter (LoRA r8):** be encouraged to use organic fertilizers. However, the use of organic fertilizers is not always easy to achieve. The aim of this study was to determine the effect of organic fertilizers on the soil fertility of a wheat field
- **agriculture adapter (LoRA r8):** avoid over-fertilization. Which of the following fertilizers is suitable for farmers to use? Nitrogen fertilizer: Nitrogen is a key nutrient for plant growth, and farmers should avoid over-fertilization. Phosphorus

## Prompt: "The city of Accra is"

- **base model:** a city of 1.5 million people. The city is located in the middle of the Ghanaian capital Accra, which is the largest city in Ghana. Accra is the capital of Ghana and the largest city in
- **health adapter (LoRA r8):** a major centre of the West African region. The city is a major centre of the West African region. The city is a major centre of the West African region. The city is a major centre of the West African region.
- **agriculture adapter (LoRA r8):** known for its vibrant culture and delicious cuisine. The city is home to several cultural festivals and events throughout the year, including the Accra International Film Festival, the Accra International Food Festival, and the Accra International Music Festival
