# Text extraction

- Repository: `NLP`
- Type of Challenge: `use case`
- Duration: `9 days`
- Deadline: `05/03/2021`
- Deployment strategy :
	 Github page
	 ppt (or alternative) presentation
- Team challenge : `4 people`

## Mission objectives 

- to preprocess textual data
- to design a text processing pipeline
- to apply NLP techniques to improve performance
- to show creative solutions on how to process unstructured data


## The Mission

[ML6](https://ml6.eu/) is a data science company handling complex cases in the world of machine learning and deep learning. Their latest point of focus has been NLP and general text-processing solutions for various companies.
Now, they want you to pose as their new intern team in charge of designing a text processing pipeline able to **extract and structure data** from a series of **similarly ordered pdf documents**.

The documents have information that is structured to be understood as a human, but not as a machine. The text inside is presented as series of **key-value pairs** with **multiple levels of hierarchy**. 
Your task is to extract this data and **produce a json file containing a dictionary-like datastructure containing this info**. You will make a **pipeline** so this process can be repeated for similar documents.
The pdf files can be found in the **'data' folder** in the same directory as this readme.


### Additional dataset info

An example of raw text extracted from such a document may look like this:

RAW:
"""
Page 1/11
Safety Data Sheet (SDS)
OSHA HazCom Standard 29 CFR 1910.1200(g) and GHS Rev 03.
Issue date 02/09/2017 Reviewed on 02/09/2017
44.2.1
* 1 Identification
· Product Identifier
· Trade name: 10N Sodium Hydroxide (NaOH 40%)
· Product Number: NGT-10N NaOH
· Relevant identified uses of the substance or mixture and uses advised against:
No further relevant information available.
· Product Description PC21 Laboratory chemicals
· Application of the substance / the mixture: Laboratory chemicals
· Details of the Supplier of the Safety Data Sheet:
· Manufacturer/Supplier:
NuGeneration Technologies, LLC (dba NuGenTec)
1155 Park Avenue, Emeryville, CA 94608
salesteam@nugentec.com www.nugentec.com
1-888-996-8436 or 1-707-820-4080 for product information
· Emergency telephone number:
PERS Emergency Response: Domestic and Canada - 1-800-633-8253, International 1-801-629-0667
"""

While the processed version retains this information:

PROCESSED:
{"identification":
	{"product_identifier":"",
	"trade_name": "NaOH 40%",
	"product_number": "NGT-10N NaOH",
	"uses_advised_against": [],
	"product_description": "PC21 Laboratory chemicals",
	"application": "Laboratory chemicals",
	"supplier_details": "",
	"manufacturer": "NuGenTec",
	"emergency_telephone_number":{
		"domestic": "1-800-633-8253",
		"international": "1-801-629-0667"
		}
	}
}

As you can see, irrelevant info is thrown away, and hierchical order is kept.

### Must-have features
1. Your pipeline needs to be able to accept ordered pdf's and return **valid json files**
2. The processed json must be **stripped of irrelevant info** (noise)


### Nice-to-have features
1. The resulting json **retains the data hierarchy**
2. The resulting json does not contain keys referring to the same information
3. Inclusion of NLP techniques to help **classify similar keys** into the same class


## Deliverables
0. A shared repository that has been **committed to daily**
1. A pipeline written in python either available as a jupyter notebook or a python project accepting pdf files and returning **valid json files**
2. A presentation showing the different steps of text processing
3. An example of one of the processed pdf's
4. An analysis of points of improvement or possible future work

### Steps
1. Analyze the documents with your group
2. Discuss and plan your schedule for the project
3. Discuss ideas on how to tackle the different objectives of the assignment
4. Design the different stages of the pipeline for the MVP
4. Create the MVP. Take care not to overcomplicate this, keep it simple (simple preprocessing, regular expressions,...)
5. Test the MVP, produce a valid json file
6. Redesign/extend the pipeline to improve performance (NLP preprocessing, classification,...)
7. Test your final pipeline
8. Create your deliverables


## Evaluation criterias
1. You have an MVP with the must have features
2. You tried to innovate using NLP or other techniques
3. You communicated clearly towards your team, and cooperated througout this excercise
4. You have a clear and shared repository that has been **committed to daily**

## A final note of encouragement

![You've got this!](https://media.giphy.com/media/yoJC2K6rCzwNY2EngA/giphy.gif)
