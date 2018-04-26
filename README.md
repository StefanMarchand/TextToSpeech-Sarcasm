# TextToSpeech-Sarcasm
Spring 2018 - Repository for Team Keep It Appropriate
Developer: Jonathan Esquivel, Stefan Marchand, Luis Sanchez, Cortlan

How to run the code:
    Warning!: Be sure to use a virtual environment to setup tensorflow, and pydub for the project. 
    Else, you might bet a permission error at runtime. Which wont let the program execute. 
    
    Project 100% tested using PyCharm 2017.3. Since we weren't able to setup the environment in anaconda. Execute main.py and follow instructions.
        Type anything and enjoy our narator. 

Packages needed
	- Tensorflow  1.2.1+
	- Pickle      12.1+
	- pydub       0.12.0+
	- numpy       1.13.1+
	- PyAudio     0.2.11
	
	Pydub
		- Can be installed from http://pydub.com/ or using "pip install pydub"  
		  or (https://github.com/jiaaro/pydub#installation) for other options
    
    Pydub also needs (libav or ffmpeg) to be installed to be able to run

        Mac (using homebrew):
            # libav
            brew install libav --with-libvorbis --with-sdl --with-theora
            ####    OR    #####
            # ffmpeg
            brew install ffmpeg --with-libvorbis --with-sdl2 --with-theora
        
        Linux (using aptitude):
            # libav
            apt-get install libav-tools libavcodec-extra-53
            ####    OR    #####
            # ffmpeg
            apt-get install ffmpeg libavcodec-extra-53
            
        Windows:
            Download and extract libav from Windows binaries provided here (http://builds.libav.org/windows/).
            Add the libav /bin folder to your PATH envvar
            pip install pydub
	
  To set up PyCharm, go to file -> settings -> project -> project: interpreter -> click (+) on bottom right
  and install the packages listed above. 
  
There shouldn't be a need to relaunch the tensorflow code in TFLanguageTranslation.py
The model has already been proccessed by us and saved using Pickle on file params.p and preprocess.p

Download Original Repository https://github.com/Cory-Edgett/IntroToAI-Text-to-Speech.git

Download and unzip.
execute main.py        (Theres no need for arguments)

After execution, you will be prompt to type any sentence you will like to say. 
Type and press enter.

If you want to exit. Type "exit"
or use Ctrl-C



What does not work?

	- We don't check for non-alphabetic character. Don't try it program will crash!
	- On program execution we get a runtime warning for saving/opening binary music files. (Not needed)
	  Don't worry the program runs even better with the warning. :)
	- Were not able to implement sarcasm into the text-to-speech
  - Were not able to run the program in spyder due to a permissions errors. Couldn't figure it out.
    
    
References:
    Used as a skeleton for our tensorflow model.
    https://github.com/Piasy/Udacity-DLND/blob/master/language-translation/dlnd_language_translation.ipynb    
    
     



	
	
	
