# med_contraindication_app
Holds code to deploy app which checks contraindicated medications for specified illnesses.

The purpose of this app is to aid the average user with minimal medical knowledge in making decisions for their health. They can type in an illness and see medications which they should double check wiith their doctor before taking.

A little bit of backstory on how this idea came about, I was having an esophagitis attack at 10pm in the night, and I was willing to try any medication we had to make it stop, but I could not because I was not sure what would adversely affect me, and I wasn't really in the best state to type something like "I have esophagitis, what can I take?" and then search through different sites and hope for a reliable source.

This app uses spaCy to parse information from the OpenFDA Api, particularly the warnings and contraindications section of medications to determine if it cannot be used when someone has a specific illness. It also allows for minor spelling errors because who is actually watching their grammar and spelling when they are actively in pain.

Overall, I hope this app can be of at least a little bit of use to the average person, even if it's just to get them to ask questions at their next routine checkup. The healthcare system can sometimes feel inaccessible and both doctors and patients do not have all the time in the world to ponder every possibility so this app is to eliminate at least one of those possibilities.

Disclaimer: This information cannot substitute advice from a trained medical professional.

Attached below is a link to the app:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://medcontraindicationapp-cw23vsc8c8kscrmeizyjwx.streamlit.app)
