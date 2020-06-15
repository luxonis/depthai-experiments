export const makeAction = (type) => {
    return (payload) => ({
        type,
        payload,
    });
};
