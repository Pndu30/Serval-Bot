# bot.py
import discord
from discord.ext import commands
from discord import Interaction
import GPTCode
from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv("DISCORD_TOKEN")

client = commands.Bot(command_prefix="sv.",intents=discord.Intents.all())

@client.event
async def on_ready():
    await client.tree.sync()
    await client.change_presence(activity=discord.activity.Game(name="UwU"),status=discord.Status.idle)
    print(f"{client.user.name} is now online")

@client.event
async def on_disconnect():
    print("f{client.user.name} is now offline")
    
@client.command()
async def intro(ctx):
    await ctx.send("Hello, my name is Serval. My job is to assist you, nice to meet you all")

@client.command(name='ask', description="Chat with the AI")
async def ask(ctx,*args):
    await ctx.send(GPTCode.get_message("-".join(args)))
    
@client.command(name="ping", description="shows the ping")
async def ping(interaction: Interaction):
    bot_latency = round(client.latency*1000)
    await interaction.response.send_message(f"Pong (Latency = {bot_latency} ms)")

# @client.command(name='help', description="")
# async def ask(ctx,*args):
#     await ctx.send(GPTCode.get_message("-".join(args)))


client.run(token)