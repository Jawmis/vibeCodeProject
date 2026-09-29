"""MovieLens ratings dashboard built with Streamlit and Plotly Express."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


DATA_PATH = Path(__file__).with_name("movie_ratings.csv")


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load and clean the movie-level ratings data once per file path."""
    data = pd.read_csv(path)

    # Accept the snake_case identifier names present in some MovieLens exports while
    # keeping the canonical names used by the dashboard and documented schema.
    data = data.rename(columns={"user_id": "userId", "movie_id": "movieId"})

    # Strip whitespace from text fields so grouping does not create duplicate labels.
    text_columns = data.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        data[column] = data[column].astype("string").str.strip()

    # Coerce malformed years to missing, remove them, then use a compact integer type.
    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data = data.dropna(subset=["year"]).copy()
    data["year"] = data["year"].astype(int)

    # Empty genres cannot contribute to genre charts, so exclude those rows up front.
    data = data[data["genres"].notna() & data["genres"].ne("")].copy()
    return data


def explode_genres(data: pd.DataFrame) -> pd.DataFrame:
    """Return a genre-level view while leaving the movie-level frame unchanged."""
    genre_data = data.copy()
    # Exploding only this analysis view duplicates ratings once per genre for genre metrics.
    genre_data["genre"] = genre_data["genres"].str.split("|")
    genre_data = genre_data.explode("genre", ignore_index=True)
    genre_data["genre"] = genre_data["genre"].astype("string").str.strip()
    return genre_data[genre_data["genre"].notna() & genre_data["genre"].ne("")]


def main() -> None:
    st.set_page_config(page_title="MovieLens Ratings Dashboard", layout="wide")
    st.title("MovieLens Ratings Dashboard")
    st.caption("Explore rating patterns by movie release year and genre.")

    try:
        movie_data = load_data(str(DATA_PATH))
    except FileNotFoundError:
        st.error(f"Could not find the dataset at `{DATA_PATH}`.")
        st.stop()

    if movie_data.empty:
        st.error("The dataset has no rows after cleaning.")
        st.stop()

    min_year = int(movie_data["year"].min())
    max_year = int(movie_data["year"].max())

    st.sidebar.header("Filters")
    year_range = st.sidebar.slider(
        "Release Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1,
    )
    rating_floor = st.sidebar.slider(
        "Minimum Rating Floor",
        min_value=10,
        max_value=200,
        value=50,
        step=5,
    )

    # Apply the year filter before exploding to keep the genre analysis view smaller.
    filtered_movies = movie_data[
        movie_data["year"].between(year_range[0], year_range[1])
    ].copy()
    filtered_genres = explode_genres(filtered_movies)

    if filtered_genres.empty:
        st.warning("No genre records match the selected year range.")
        st.stop()

    st.subheader("Genre Breakdown")
    genre_counts = (
        filtered_genres.groupby("genre", as_index=False)
        .size()
        .rename(columns={"size": "rating_count"})
        .sort_values("rating_count", ascending=False)
    )
    genre_count_chart = px.bar(
        genre_counts,
        x="genre",
        y="rating_count",
        title="Rating Frequency by Genre",
        labels={"genre": "Genre", "rating_count": "Number of Ratings"},
        category_orders={"genre": genre_counts["genre"].tolist()},
    )
    genre_count_chart.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(genre_count_chart, use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Genre Satisfaction")
        genre_means = (
            filtered_genres.groupby("genre", as_index=False)["rating"]
            .mean()
            .rename(columns={"rating": "mean_rating"})
            .sort_values("mean_rating", ascending=True)
        )
        genre_mean_chart = px.bar(
            genre_means,
            x="mean_rating",
            y="genre",
            orientation="h",
            title="Mean Rating by Genre",
            labels={"genre": "Genre", "mean_rating": "Mean Rating"},
            category_orders={"genre": genre_means["genre"].tolist()},
            text=genre_means["mean_rating"].map(lambda value: f"{value:.2f}"),
        )
        genre_mean_chart.update_traces(textposition="outside", cliponaxis=False)
        genre_mean_chart.update_xaxes(tickformat=".2f")
        st.plotly_chart(genre_mean_chart, use_container_width=True)

    with col_right:
        st.subheader("Ratings Over Time")
        yearly_means = (
            filtered_movies.groupby("year", as_index=False)["rating"]
            .mean()
            .rename(columns={"rating": "mean_rating"})
            .sort_values("year")
        )
        yearly_chart = px.line(
            yearly_means,
            x="year",
            y="mean_rating",
            markers=True,
            title="Mean Rating by Movie Release Year",
            labels={"year": "Release Year", "mean_rating": "Mean Rating"},
        )
        yearly_chart.update_yaxes(tickformat=".2f")
        st.plotly_chart(yearly_chart, use_container_width=True)

    st.subheader("Top 5 Movies by Mean Rating")
    # This aggregation uses all cleaned rows, deliberately independent of the year filter.
    movie_summary = (
        movie_data.groupby(["movieId", "title"], as_index=False)
        .agg(rating_count=("rating", "size"), mean_rating=("rating", "mean"))
    )
    top_movies = (
        movie_summary[movie_summary["rating_count"] >= rating_floor]
        .sort_values(["mean_rating", "rating_count", "title"], ascending=[False, False, True])
        .head(5)
        .sort_values("mean_rating", ascending=True)
    )

    if top_movies.empty:
        st.info("No movies meet the selected minimum rating floor.")
    else:
        top_movie_chart = px.bar(
            top_movies,
            x="mean_rating",
            y="title",
            orientation="h",
            title=f"Top 5 Movies with at Least {rating_floor} Ratings",
            labels={"title": "Movie", "mean_rating": "Mean Rating"},
            category_orders={"title": top_movies["title"].tolist()},
            text=top_movies["mean_rating"].map(lambda value: f"{value:.2f}"),
            hover_data={"rating_count": True, "mean_rating": ":.2f"},
        )
        top_movie_chart.update_traces(textposition="outside", cliponaxis=False)
        top_movie_chart.update_xaxes(tickformat=".2f")
        st.plotly_chart(top_movie_chart, use_container_width=True)


if __name__ == "__main__":
    main()
