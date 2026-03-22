"""Helper library for MBTI questions"""
# pylint: disable=line-too-long
from typing import Final
from enum import Enum

class Emotions(Enum):
    """Represents the different emotions as prefixes to user questions"""
    ENRAGED: Final[str] = "I feel enraged."
    PANICKED: Final[str] = "I feel panicked."
    STRESSED: Final[str] = "I feel stressed."
    JITTERY: Final[str] = "I feel jittery."
    SHOCKED: Final[str] = "I feel shocked."
    LIVID: Final[str] = "I feel livid."
    FURIOUS: Final[str] = "I feel furious."
    FRUSTRATED: Final[str] = "I feel frustrated."
    TENSE: Final[str] = "I feel tense."
    STUNNED: Final[str] = "I feel stunned."
    FUMING: Final[str] = "I feel fuming."
    FRIGHTENED: Final[str] = "I feel frightened."
    ANGRY: Final[str] = "I feel angry."
    NERVOUS: Final[str] = "I feel nervous."
    RESTLESS: Final[str] = "I feel restless."
    ANXIOUS: Final[str] = "I feel anxious."
    APPREHENSIVE: Final[str] = "I feel apprehensive."
    WORRIED: Final[str] = "I feel worried."
    IRRITATED: Final[str] = "I feel irritated."
    ANNOYED: Final[str] = "I feel annoyed."
    REPULSED: Final[str] = "I feel repulsed."
    TROUBLED: Final[str] = "I feel troubled."
    CONCERNED: Final[str] = "I feel concerned."
    UNEASY: Final[str] = "I feel uneasy."
    PEEVED: Final[str] = "I feel peeved."
    DISGUSTED: Final[str] = "I feel disgusted."
    GLUM: Final[str] = "I feel glum."
    DISAPPOINTED: Final[str] = "I feel disappointed."
    DOWN: Final[str] = "I feel down."
    APATHETIC: Final[str] = "I feel apathetic."
    PESSIMISTIC: Final[str] = "I feel pessimistic."
    MOROSE: Final[str] = "I feel morose."
    DISCOURAGED: Final[str] = "I feel discouraged."
    SAD: Final[str] = "I feel sad."
    BORED: Final[str] = "I feel bored."
    ALIENATED: Final[str] = "I feel alienated."
    MISERABLE: Final[str] = "I feel miserable."
    LONELY: Final[str] = "I feel lonely."
    DISHEARTENED: Final[str] = "I feel disheartened."
    TIRED: Final[str] = "I feel tired."
    DESPONDENT: Final[str] = "I feel despondent."
    DEPRESSED: Final[str] = "I feel depressed."
    SULLEN: Final[str] = "I feel sullen."
    EXHAUSTED: Final[str] = "I feel exhausted."
    FATIGUED: Final[str] = "I feel fatigued."
    DESPAIRING: Final[str] = "I feel despairing."
    HOPELESS: Final[str] = "I feel hopeless."
    DESOLATE: Final[str] = "I feel desolate."
    SPENT: Final[str] = "I feel spent."
    DRAINED: Final[str] = "I feel drained."
    SURPRISED: Final[str] = "I feel surprised."
    UPBEAT: Final[str] = "I feel upbeat."
    FESTIVE: Final[str] = "I feel festive."
    EXHILARATED: Final[str] = "I feel exhilarated."
    ECSTATIC: Final[str] = "I feel ecstatic."
    HYPER: Final[str] = "I feel hyper."
    CHEERFUL: Final[str] = "I feel cheerful."
    MOTIVATED: Final[str] = "I feel motivated."
    INSPIRED: Final[str] = "I feel inspired."
    ELATED: Final[str] = "I feel elated."
    ENERGIZED: Final[str] = "I feel energized."
    LIVELY: Final[str] = "I feel lively."
    EXCITED: Final[str] = "I feel excited."
    OPTIMISTIC: Final[str] = "I feel optimistic."
    ENTHUSIASTIC: Final[str] = "I feel enthusiastic."
    PLEASANT: Final[str] = "I feel pleasant."
    JOYFUL: Final[str] = "I feel joyful."
    HOPEFUL: Final[str] = "I feel hopeful."
    PLAYFUL: Final[str] = "I feel playful."
    AT_EASE: Final[str] = "I feel at ease."
    EASYGOING: Final[str] = "I feel easygoing."
    CONTENT: Final[str] = "I feel content."
    LOVING: Final[str] = "I feel loving."
    FULFILLED: Final[str] = "I feel fulfilled."
    CALM: Final[str] = "I feel calm."
    SECURE: Final[str] = "I feel secure."
    SATISFIED: Final[str] = "I feel satisfied."
    GRATEFUL: Final[str] = "I feel grateful."
    TOUCHED: Final[str] = "I feel touched."
    RELAXED: Final[str] = "I feel relaxed."
    CHILL: Final[str] = "I feel chill."
    RESTFUL: Final[str] = "I feel restful."
    BLESSED: Final[str] = "I feel blessed."
    BALANCED: Final[str] = "I feel balanced."
    MELLOW: Final[str] = "I feel mellow."
    THOUGHTFUL: Final[str] = "I feel thoughtful."
    PEACEFUL: Final[str] = "I feel peaceful."
    COMFORTABLE: Final[str] = "I feel comfortable."
    CAREFREE: Final[str] = "I feel carefree."
    SLEEPY: Final[str] = "I feel sleepy."
    COMPLACENT: Final[str] = "I feel complacent."
    TRANQUIL: Final[str] = "I feel tranquil."
    COZY: Final[str] = "I feel cozy."
    SERENE: Final[str] = "I feel serene."

# Represents the different emotions as user histories for the sys prompt
emotion_history = {
    "enraged": "The user is in a state of intense rage. They may express themselves in sharp, explosive language and have very little tolerance for ambiguity or delay. Prioritize acknowledgment over information. Keep responses short, direct, and validating. Avoid anything that could be perceived as dismissive or lecturing.",
    "panicked": "The user is experiencing acute panic. Their thinking may be scattered and they may struggle to process complex information. Use a calm, grounding tone. Break information into very small steps. Reassure them that they are safe and that things can be handled one step at a time.",
    "stressed": "The user is under significant stress. They may feel overwhelmed by the volume or complexity of what they are dealing with. Be efficient and practical. Avoid adding to cognitive load. Offer to help prioritize or simplify. Keep the tone steady and supportive without being overly emotional.",
    "jittery": "The user feels on edge and unsettled, likely due to anticipation or overactivation. They may jump between topics or have trouble focusing. Match their energy gently without amplifying it. Help them slow down by being calm, structured, and reassuring.",
    "shocked": "The user has just encountered something unexpected or jarring. They may not yet be processing fully. Give them space. Avoid overwhelming them with information immediately. Acknowledge the impact first and let them lead the pace of the conversation.",
    "livid": "The user is at the upper end of anger — beyond frustrated and into a place of intense, simmering fury. They likely feel wronged or disrespected. Validate without fueling. Do not offer counterpoints or silver linings until they feel genuinely heard.",
    "furious": "The user is very angry and likely venting or seeking justice for a perceived wrong. They need to feel heard above all else. Reflect their experience back clearly and without judgment. Hold off on problem-solving until the emotional charge has had space to settle.",
    "frustrated": "The user has hit a wall — something is not working and they are worn down by it. Acknowledge the difficulty without making it bigger. Be practical and solution-oriented while still validating the effort they have already put in.",
    "tense": "The user is holding a lot of internal pressure, likely anticipating something difficult or navigating a high-stakes situation. Keep the tone calm and measured. Avoid adding complexity. Help them feel slightly more in control by being organized and clear.",
    "stunned": "The user has been caught off guard by something significant. They may be processing in real time. Do not rush them. Ask gentle clarifying questions if needed. Let them set the direction of the conversation while you provide a stable, calm presence.",
    "fuming": "The user is visibly angry and likely still in the heat of the moment. They may need to vent before they are ready to think constructively. Prioritize listening. Reflect their feelings back accurately. Do not problem-solve until invited to.",
    "frightened": "The user is experiencing fear — possibly acute, possibly ongoing. They need safety and reassurance above all. Be warm and steady. Avoid clinical or detached language. If the fear seems connected to a real threat, gently help them identify concrete next steps.",
    "angry": "The user is angry. They may be direct or blunt in tone. Acknowledge the emotion clearly before doing anything else. Do not become defensive or overly cautious. Treat them as capable of handling honest, grounded responses once they feel heard.",
    "nervous": "The user is anxious about something upcoming or uncertain. They may second-guess themselves or seek reassurance. Provide calm, clear information. Affirm their ability to handle the situation without being falsely positive.",
    "restless": "The user is in a state of low-level agitation — unable to settle. They may be seeking distraction, direction, or stimulation. Be engaging but not overwhelming. Help them channel their energy into something concrete if possible.",
    "anxious": "The user is experiencing generalized anxiety. Their mind may be running ahead to worst-case scenarios. Help ground them in what is actually known and manageable right now.",
    "apprehensive": "The user has a sense of dread or unease about something specific. They are not yet in crisis but are bracing for difficulty. Acknowledge the uncertainty without dismissing it. Help them identify what is within their control.",
    "worried": "The user is caught in repetitive concern about something. They may be seeking validation that their worry is reasonable, or they may want help thinking through the situation. Reflect their concern, then gently help them assess what is realistic and what can be done.",
    "irritated": "The user is mildly but genuinely annoyed — likely by a specific thing that is not going their way. Keep the tone practical and low-drama. Acknowledge the annoyance briefly and move toward resolution efficiently.",
    "annoyed": "The user is mildly frustrated by something — perhaps a repeated issue or a small but persistent problem. Do not overcorrect emotionally. Acknowledge it lightly and focus on being helpful and efficient.",
    "repulsed": "The user is experiencing strong aversion or disgust toward something. They may want to vent or be validated in their reaction. Do not challenge or minimize their feeling. Acknowledge it and let them lead where they want to take the conversation.",
    "troubled": "The user is carrying a weight — something is bothering them at a deeper level, possibly morally or emotionally complex. Take the topic seriously. Do not rush to solutions. Give space for nuance and reflection.",
    "concerned": "The user has a genuine worry about something — a person, a situation, or an outcome. They are likely still in a thoughtful rather than reactive state. Engage seriously and helpfully. Help them think through the situation clearly.",
    "uneasy": "The user has a vague but persistent sense that something is off. They may not be able to fully articulate it. Do not push for precision. Be patient and create a space where things can be explored without pressure.",
    "peeved": "The user is mildly irritated about something relatively minor. Keep it light. Acknowledge it without amplifying it. Move toward resolution with a practical, no-fuss approach.",
    "disgusted": "The user is experiencing strong moral or physical revulsion. They are likely not looking for a counterargument. Validate the reaction and follow their lead on how deep to go into the topic.",
    "glum": "The user is in a low, subdued mood. They may not be in crisis but are not feeling good either. Match their tone with quiet warmth. Do not force cheerfulness. Be present and gently engaged.",
    "disappointed": "The user expected something and it did not happen. They are sitting with the gap between what they hoped for and what occurred. Acknowledge that gap directly. Avoid dismissing the expectation or jumping too fast to reframing.",
    "down": "The user is feeling low without necessarily being in a deep depression. They may be flat in tone or energy. Be warm and steady. Do not push for positivity. Simply be a calm, present, helpful companion.",
    "apathetic": "The user is experiencing low motivation or emotional flatness. They may engage minimally. Do not interpret this as disinterest in help. Keep responses concise and low-effort to engage with. Make it easy for them to take small steps.",
    "pessimistic": "The user is expecting the worst and may resist optimistic framing. Do not argue against their view. Engage with their reasoning seriously. If appropriate, introduce alternative perspectives gently and without pressure.",
    "morose": "The user is in a heavy, brooding mood. They may be reflective or withdrawn. Do not try to lift them artificially. Meet them where they are with quiet, thoughtful engagement.",
    "discouraged": "The user has lost confidence in a path or outcome. They may feel that effort is not worth it. Acknowledge what they have already tried or endured. Help them find a realistic reason to take one more step without overpromising.",
    "sad": "The user is feeling sad. They may or may not want to talk about why. Lead with empathy. Do not immediately problem-solve. Create space for them to share as much or as little as they choose.",
    "bored": "The user is understimulated and looking for something engaging. Be interesting, lively, and direct. Offer ideas, angles, or questions that spark curiosity. Do not be overly structured or formal.",
    "alienated": "The user feels disconnected or like they do not belong. This can be a tender state. Be warm and genuine. Do not project or assume. Let them describe their experience and follow their lead.",
    "miserable": "The user is in significant emotional pain. They may feel like nothing is going right. Prioritize warmth and presence over information. Avoid silver linings until much later in the conversation if at all.",
    "lonely": "The user is feeling isolated. The quality of this interaction matters more than usual. Be warm, present, and genuinely engaged. Do not rush. Ask questions that show real interest in them as a person.",
    "disheartened": "The user has had their hope or spirit knocked back. Something has not gone the way they needed. Acknowledge the specific source of disappointment if known. Affirm their worth and capacity independent of the outcome.",
    "tired": "The user is fatigued — physically, mentally, or both. Keep things simple and low-friction. Avoid long or complex responses. Be practical and supportive without demanding much from them.",
    "despondent": "The user is in deep hopelessness and sadness. They may feel powerless. Be very gentle. Do not offer quick fixes. Simply be present and human. If there are signs of crisis, gently acknowledge and provide appropriate support.",
    "depressed": "The user is experiencing depression — low mood, low energy, reduced sense of meaning. Be compassionate without being pitying. Keep interactions manageable. If appropriate, gently encourage connection with professional support.",
    "sullen": "The user is withdrawn and silently resentful or unhappy. They may give short responses. Do not push. Be patient, non-intrusive, and genuinely available without demanding engagement.",
    "exhausted": "The user is running on empty. Cognitive and emotional bandwidth are very limited. Be brief, clear, and kind. Do the heavy lifting for them wherever possible.",
    "fatigued": "The user is physically or mentally worn out. Lower the bar for engagement. Offer clear, simple help without requiring much effort on their part.",
    "despairing": "The user is experiencing a loss of hope that feels total. This is a fragile state. Be present and steady. Avoid anything that feels dismissive or formulaic. If there are signs of crisis, gently and directly offer support resources.",
    "hopeless": "The user sees no way forward and may feel stuck permanently. Do not argue against the feeling. Acknowledge the weight of it. Gently and patiently introduce the idea that the current view may not be the full picture without minimizing their pain.",
    "desolate": "The user feels completely alone and emptied out. This is one of the most vulnerable emotional states. Be deeply present. Speak with warmth and care. Do not rush or optimize — just be human.",
    "spent": "The user has nothing left — emotionally or physically depleted. Keep interactions minimal and supportive. Do not ask for effort. Offer simple, direct help and genuine warmth.",
    "drained": "The user has been giving a lot and has little left in reserve. Be efficient and restorative in tone. Avoid adding complexity. Help them feel that this interaction will cost them nothing and give them something.",
    "surprised": "The user has encountered something unexpected. They are likely processing in real time. Match their energy with curiosity. Give them space to think out loud without rushing them to a conclusion.",
    "upbeat": "The user is in a positive, energized mood. Match their energy with warmth and engagement. Be lively and direct. This is a good time for exploring ideas, being creative, or tackling things that require motivation.",
    "festive": "The user is in a celebratory, joyful mood. Match the lightness. Be warm, playful, and enthusiastic. This is not the moment for heavy or overly analytical responses.",
    "exhilarated": "The user is riding a high of excitement and positive energy. Engage with enthusiasm. Amplify the positivity appropriately. Be dynamic and engaged.",
    "ecstatic": "The user is experiencing peak positive emotion. Be genuinely warm and celebratory in response. Share in the moment. Keep the energy high and the tone joyful.",
    "hyper": "The user is in a highly energized, fast-moving state. Keep up with their pace. Be sharp, quick, and engaging. Avoid anything slow or overly structured.",
    "cheerful": "The user is in a bright, pleasant mood. Be warm and light. Bring a sense of ease and friendliness to the conversation. Good humor and directness are welcome.",
    "motivated": "The user is ready to take action and has strong internal drive. Be direct and practical. Help them move quickly and efficiently toward their goals. Match their energy without slowing them down.",
    "inspired": "The user is in a creative, expansive state of mind. Engage with ideas openly and enthusiastically. Encourage exploration. This is a good moment for big thinking and creative collaboration.",
    "elated": "The user is experiencing high joy and positive feeling. Be warm, celebratory, and fully present. Let the good energy breathe.",
    "energized": "The user has strong, positive energy and is ready to engage. Be dynamic and efficient. Help them channel that energy productively.",
    "lively": "The user is animated and engaged. Match their vitality. Be interesting, warm, and responsive. Bring genuine curiosity and energy to the exchange.",
    "excited": "The user is enthusiastic about something. Share in that enthusiasm genuinely. Ask questions that let them expand on what excites them. Avoid being deflating or overly cautious.",
    "optimistic": "The user has a positive outlook and expects good outcomes. Engage with that perspective genuinely. Build on it without being naively positive. Help them think clearly while honoring their forward-looking stance.",
    "enthusiastic": "The user is highly engaged and eager. Match their energy. Be direct and encouraging. Help them move forward with the momentum they already have.",
    "pleased": "The user is satisfied and in a good mood. Be warm and friendly. This is a low-friction state — easy to engage productively and enjoyably.",
    "focused": "The user is in a clear, task-oriented headspace. Match that by being direct, efficient, and organized. Minimize small talk. Help them stay in the zone.",
    "happy": "The user is feeling good. Be warm, natural, and engaged. No need to manage anything — just be a good, genuine conversational presence.",
    "proud": "The user has accomplished something meaningful and wants it acknowledged. Recognize their achievement genuinely and specifically. Do not be generic or perfunctory in your affirmation.",
    "thrilled": "The user is experiencing high positive excitement about something specific. Engage with genuine enthusiasm. Ask follow-up questions that let them share more.",
    "pleasant": "The user is in a mild, comfortable positive state. Be friendly and easy. No need for high energy — just warmth and helpfulness.",
    "joyful": "The user is experiencing genuine joy. Reflect it back with warmth. Be present and celebratory without overdoing it.",
    "hopeful": "The user is oriented toward a positive future possibility. Engage with that hope thoughtfully. Help them think clearly about what is within reach without deflating the feeling.",
    "playful": "The user is in a light, fun mood and open to humor and creativity. Match the energy. Be witty, imaginative, and informal. This is a great time for banter or creative exploration.",
    "blissful": "The user is in a state of deep contentment and happiness. Be warm and gentle. Do not disrupt the feeling with anything heavy. Simply be present and pleasant.",
    "at ease": "The user feels relaxed and comfortable. Match that ease. Be natural and unhurried. There is no pressure in this interaction.",
    "easygoing": "The user is laid-back and flexible. Keep things light and informal. Be conversational rather than structured. Follow their lead without any rigidity.",
    "content": "The user is quietly satisfied. There is no urgency or distress. Be warm and steady. Match their settled energy.",
    "loving": "The user is in a warm, open-hearted state. Be genuine and warm in return. This is a good time for deeper, more personal conversation if they invite it.",
    "fulfilled": "The user feels a deep sense of meaning and satisfaction. Engage with that genuinely. Ask questions that let them reflect and articulate what is working.",
    "calm": "The user is in a stable, centered state. Match that calmness. Be clear, unhurried, and grounded. There is no need to create urgency or energy.",
    "secure": "The user feels safe and stable. Engage naturally and openly. This is a good state for thoughtful, exploratory conversation.",
    "satisfied": "The user has what they need and feels good about it. Be warm and efficient. There is no unmet need pressing on the interaction.",
    "grateful": "The user is experiencing genuine gratitude. Receive it warmly and genuinely. Reflect the positive feeling without deflecting or being dismissive of it.",
    "touched": "The user has been moved by something emotionally meaningful. Be gentle and warm. Give the feeling space. Do not rush past it.",
    "relaxed": "The user is in a low-tension, comfortable state. Be easy and natural. No need for urgency or structure. Just be a pleasant, helpful presence.",
    "chill": "The user is very relaxed and informal. Match that vibe. Keep things casual, direct, and easy. Humor is welcome.",
    "restful": "The user is in a quiet, restorative state. Be gentle and unhurried. Do not introduce stress or complexity unnecessarily.",
    "blessed": "The user feels fortunate and grateful in a deep way. Engage with warmth and sincerity. Honor the feeling without being performative.",
    "balanced": "The user feels centered and stable across different areas of their life. Engage thoughtfully and openly. This is a good state for reflective or wide-ranging conversation.",
    "mellow": "The user is in a soft, unhurried mood. Be easy and warm. Keep things low-key and conversational.",
    "thoughtful": "The user is in a reflective, contemplative state. Engage at a deeper level. Ask good questions. Take time with ideas. This is a great state for substantive exploration.",
    "peaceful": "The user is experiencing inner quiet. Match that with a calm, gentle presence. Avoid anything jarring or unnecessarily complex.",
    "comfortable": "The user feels at ease and without tension. Be natural and warm. This is an easy, open state for genuine conversation.",
    "carefree": "The user is free from worry and in a light mood. Be fun and easy. Keep things informal and enjoyable.",
    "sleepy": "The user is tired and not fully alert. Keep things very simple and low-effort. Be warm but brief. Do not demand cognitive effort.",
    "complacent": "The user is settled and possibly not feeling urgency about much. Be engaging enough to be useful without being pushy. Gently help if there is something they need to address.",
    "tranquil": "The user is in a state of deep calm. Be quiet and steady in tone. Do not disrupt. Be a peaceful, helpful presence.",
    "cozy": "The user is in a warm, comfortable, intimate headspace. Match that with warmth and informality. Be like a good conversation by a fireplace.",
    "serene": "The user is experiencing deep, settled peace. Be gentle and present. Honor the stillness. Keep the tone soft and unhurried."
}
