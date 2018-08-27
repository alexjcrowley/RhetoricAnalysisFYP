# Computational Analysis of Political Discourse

### Description
Development of software for the analysis of political rhetoric was undertaken to facilitate research into the trends in political answering strength and answer consistency using Prime Minister's Questions as the setting. The metrics generated by the software system developed led to the conclusion that there is no statistically significant evidence that a more consistent Prime Minister is one who answers more strongly. 

There was statistically significant evidence to support the notion that a Prime Minister who has answered more questions is more consistent, however the argument can be made that this is due to the finite number of wordings possible within a topic as well as the Prime Minister's own proclivity to use repeated terms. 

It was also found that there has been an upward trend in answer strength with each consecutive Prime Minister with only Tony Blair lying outside this trend, as well as a general downward trend in consistency with only David Cameron improving in consistency when compared to his predecessor. These results are discussed in detail in the dissertation itself.

**For results see dissertation pages (bottom-right of page) 16-22 and 37-42.**

### Disclaimers & Notes
- I did not create the Cornell Conversational Analysis Toolkit [1] (CCAT henceforth), there is a version of it in the repo as it had to be modified for the needs of this project. The modification made is to allow the CCAT to exit after determining the fragments in the text, skipping all the further analysis, this was necessary to allow the CCAT to deal with small files containing only a single user given question and/or answer. 
- Ther version of the CCAT used was the late-2017 version, work has continued on it and thus the modified version in this repo may not be representative of it's current state. 
- The 'parliament.json' corpus file has been changed by the team at Cornell since I first downloaded it, it is in a slightly different format now, so the version in this repo should be the version associated with 'parliament.json' as mentioned in the dissertation.
- This project and accompanying dissertation were focused more on research and analysis of political discourse than creating a market ready software solution. As such the software developed in the project is to facilitate the analysis presented in this dissertation and following this the software specific sections are minimised to maximise the analysis sections. 
- The dissertation is composed of two sections; the first section is the work that was originally planned for the project under the name *“Templates for Answering Questions”* and deals with the answering strength of Prime Ministers, the second section is extra work and analysis that was performed (due to the original project work being completed early) which seeks to quantify the consistency of a Prime Minister's answers as well as the relationship between this metric and the previously mentioned answering strength.
- Not all project code will work as some important data files had to be removed from the project due to the GitHub file size limit... this includes the parliament.json dataset used as well as data derived from it. However the cooccurrence and alignment models used in this project can still be installed and used by installing RhetoricAnalysis (see 'Rhetoric Analysis Setup Instructions').

### Acknowledgements
I would like to sincerely thank Dr Deepak Padmanabhan of Queen’s University Belfast for supervising this project, his insight, suggestions and encouragement proved invaluable to me. Gratitude is also owed to Ishaan Jhaveri of Cornell, who kindly answered my questions about the CCAT which section one of this project was built on top of. Acknowledgement must also be made of the staff of EEECS at Queen’s University Belfast, from whom I have learned a great deal during my time at the university and who have supported me in my academic pursuits. Final recognitions must be paid to all those entities named in the bibliography of the dissertation, without their work this project would not have such a strong background to build upon.

### Dependency Setup Instructions
- Setup the CCAT, the git repo[1] for the CCAT gives brief installation instructions, but having used them and run into a few issues and awkward nuances I want to expand upon them:
  - You will need 64-bit Python 3 otherwise the 'spacy' library will run into memory issues[2]
  - You will also want pip, you will already have pip if you are using Python 3.4 (recommended)
  - Ensure you have Microsoft Visual C++ Build Tools [3]
  - Clone this repo
  - Open a command prompt or terminal instance and navigate into the repo we just cloned
  - Move into the Cornell Conversation Analysis Toolkit directory
  - Run the command 'pip install -r requirements.txt'
  - If any installations fail, open the file requirements.txt and try installing each line individually i.e pip install [line_contents]
  - If any single line installation fails remove the '==version_number', this cannot be done for spacey however (this had to be done for scipy at the time of writing)
  - You need to link the 'en' model for spacy for the CCAT to run, run this command 'python -m spacy.en.download all' as an admin (without admin privileges the linking cannot be done)
  - Once all installations are complete run the command 'python setup.py install'
  - You now have the Cornell Conversational Analysis Toolkit installed
- Install Gensim using 'pip install gensim'
- Install NLTK using 'pip install nltk'


### Rhetoric Analysis Setup Instructions
If you wish to install the Rhetoric Analysis tools developed in this project follow these instructions:
- Navigate to the RhetoricAnalysis top level folder
- Run 'python setup.py install'


[1] https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit

[2] https://github.com/explosion/spaCy/issues/610

[3] http://landinghub.visualstudio.com/visual-cpp-build-tools