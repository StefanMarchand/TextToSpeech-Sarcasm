#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 12:56:27 2018

@author: Luis Sanchez

Purpose: Used to execute the main portion of the code
         Prompts user to enter any word, or sentence they want the computer to speak
         Later saying the result back to the user after pressing enter
"""


# from TFLanguageTranslation import batch_size
from translator import wordsToPhonetics
from Speak import sayThis


# Main execution of the code
def main():
    while(1):
        print("What can I say for you?")
        ans = input(">>> ")
        if ans == "exit":
            break

        # given any word, it translate it to phonetics
        phonetics = wordsToPhonetics(ans)

        # seperate each word by spaces, and add a delimited to space out words
        combPhonetics = []
        for phonetic in phonetics:
            # remove trailing empty space
            phonetic = phonetic.strip()
            splitted = phonetic.split(" ")
            for ph in splitted:
                combPhonetics.append(ph)

            combPhonetics.append("SPC")

        # Say the result
        sayThis(combPhonetics)



main()