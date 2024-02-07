#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("/n Welcome to the Hangman game")

import random

words = ["python", "hangman", "programming", "computer", "coding", "challenge"]
word_to_guess = random.choice(words)
guessed_letters = []
chance = 3  # Set the maximum number of incorrect attempts

while chance > 0:
    display = ""
    for letter in word_to_guess:
        if letter in guessed_letters:
            display += letter
        else:
            display += "_"

    print(display)

    if "_" not in display:
        print("Congratulations! You guessed the word!")
        break

    print(f"Attempts left: {chance}")
    guess = input("Guess a letter: ").lower()

    if len(guess) != 1 or not guess.isalpha():
        print("Please enter a single alphabetic character.")
        continue

    if guess in guessed_letters:
        print("You've already guessed that letter. Try again.")
        continue

    guessed_letters.append(guess)

    if guess not in word_to_guess:
        chance -= 1
        print(f"Incorrect! {chance} attempts left.")

if "_" in display:
    print(f"Sorry, you ran out of attempts. The word was: {word_to_guess}")

