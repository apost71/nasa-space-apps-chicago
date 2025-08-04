"""Command-line interface for the Data Explorer."""

import asyncio
import click
from dotenv import load_dotenv
from .agent import run_agent

load_dotenv()


@click.group()
def cli():
    """Data Explorer CLI for NASA AppEEARS and Elastic integration."""
    pass


@cli.command()
@click.argument("query")
def explore(query: str):
    """Run a data exploration query."""
    result = asyncio.run(run_agent(query))
    click.echo(result)


def main():
    """Entry point for the CLI."""
    cli()
