# Gelin Eguinosa Rosique

from abc import ABC, abstractmethod


class CorpusCord19(ABC):
    """
    Abstract class with the methods that each Corpus Manager in the project needs
    to implement.
    """

    @abstractmethod
    def papers_cord_uids(self):
        """
        Get the identifier (cord_uid) of the CORD-19 papers present in
        this corpus.

        Returns: A list of strings containing the 'cord_uids'.
        """
        pass

    @abstractmethod
    def paper_title_abstract(self, cord_uid):
        """
        Get the title and abstract of the paper with the given 'cord_uid'.

        Args:
            cord_uid: String with the identifier of the paper.

        Returns:
            A string containing the title and abstract of the paper.
        """
        pass

    @abstractmethod
    def paper_body_text(self, cord_uid):
        """
        Get text in the body of the given 'cord_uid' paper, which is the content
        of the paper excluding the title and abstract.

        Args:
            cord_uid: A string with the identifier of the paper.

        Returns:
            A string with the body text of the paper.
        """
        pass

    @abstractmethod
    def paper_embedding(self, cord_uid):
        """
        Get the Specter embedding of the given 'cord_uid' paper.

        Args:
            cord_uid: A string with the identifier of the paper.

        Returns:
            A List[float] containing the embedding of the paper.
        """
        pass

    def paper_content(self, cord_uid):
        """
        Get the full content of the 'cord_uid' paper.

        Args:
            cord_uid: A string with the identifier of the paper.

        Returns:
            A string containing the title, abstract and body text of the paper.
        """
        # Use paper_title_abstract() & paper_body_text()
        full_text = self.paper_title_abstract(cord_uid) + '\n\n' + self.paper_body_text(cord_uid)
        return full_text

    def all_papers_title_abstract(self):
        """
        Create an iterator of strings containing the title and abstract of all
        the papers in the current corpus.

        Returns: An iterator of strings.
        """
        for cord_uid in self.papers_cord_uids():
            yield self.paper_title_abstract(cord_uid)

    def all_papers_body_text(self):
        """
        Create an iterator containing the body text of all the papers in the
        current corpus.

        Returns: An iterator of strings.
        """
        for cord_uid in self.papers_cord_uids():
            yield self.paper_body_text(cord_uid)

    def all_papers_content(self):
        """
        Create an iterator containing the content of all the papers in the
        current CORD-19 corpus.

        Returns: An iterator of strings.
        """
        for cord_uid in self.papers_cord_uids():
            yield self.paper_content(cord_uid)

    def all_papers_embeddings(self):
        """
        Create an iterator containing the embeddings of the papers in the
        current CORD-19 corpus.

        Returns: An iterator of List[float].
        """
        for cord_uid in self.papers_cord_uids():
            yield self.paper_embedding(cord_uid)